from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
import os
import random

import config as cfg
from dataset import *
from train import *
from model import *
from style_loss import *
import matplotlib.pyplot as plt

if cfg.ENABLE_TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter


if cfg.FIXED_RANDOM:
    print("Fixing random in order to enable reproduction")
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)


print("Setting device for pytorch")
if cfg.USE_GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

print("Device: ", device)


# setting transforms
transforms_list = []
if cfg.TO_RESIZE:
    print("resizing images to", cfg.RESIZE_DIM)
    transforms_list.append(transforms.Resize((cfg.RESIZE_DIM, cfg.RESIZE_DIM)))
transforms_list.append(transforms.ToTensor())
if cfg.TO_NORMALIZE:
    transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))#(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])) #(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))


print("Creating training set")


train_set = ImagesDataset(images_dir_path=cfg.DATASET_PATH, 
                set_type=SetType.TrainSet, 
                masking_method=eval("MaskingMethod."+cfg.MASKING_METHOD), 
                image_dim_size=cfg.IMAGE_SIZE, 
                mask_dim_size=cfg.MASK_SIZE,
                mask_max_pixels=cfg.RANDOM_REGION_MASK_MAX_PIXELS,
                overlap=cfg.MASK_OVERLAP,
                transform=transforms.Compose(transforms_list))

if cfg.SHOW_IMAGE:
    print("Debug only: showing training set examples (with masks)")
    for image in train_set:
        try:
            data = image
            plt.imshow(np.transpose(data["masked_image"].numpy(), (1, 2, 0)))
            plt.show()
            plt.imshow(np.transpose(data["orig_parts"].numpy(), (1, 2, 0)))
            plt.show()
        except KeyboardInterrupt:
            exit()

print("Creating validation set")
valid_set = ImagesDataset(images_dir_path=cfg.DATASET_PATH, 
                set_type=SetType.ValidSet, 
                masking_method=eval("MaskingMethod."+cfg.MASKING_METHOD), 
                image_dim_size=cfg.IMAGE_SIZE, 
                mask_dim_size=cfg.MASK_SIZE,
                mask_max_pixels=cfg.RANDOM_REGION_MASK_MAX_PIXELS,
                overlap=cfg.MASK_OVERLAP,
                transform=transforms.Compose(transforms_list))
                

dataset_sizes = {
    'train' : len(train_set),
    'valid' : len(valid_set)
}
   
data_loaders = {
    'train': DataLoader(train_set, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=True, drop_last=True),
    'valid': DataLoader(valid_set, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=False, drop_last=True)
}

print("Creating models")

if cfg.EXTERNAL_REF_NET_CROSS_STYLE_LOSS:
    extract_features = False
else:
    if cfg.NET_CROSS_STYLE_LOSS:
        extract_features = True
    else:
        extract_features = False

if cfg.MASKING_METHOD == "CentralRegion":
    gen_model = GeneratorNet(output_full_image=False, output_size=cfg.MASK_SIZE, extract_features=extract_features) #False
    disc_model = DiscriminatorNet(input_full_image=False, input_size=cfg.MASK_SIZE) #False
else:
    gen_model = GeneratorNet(output_full_image=True, extract_features=extract_features)
    disc_model = DiscriminatorNet(input_full_image=True)

if cfg.EXTERNAL_REF_NET_CROSS_STYLE_LOSS:
    print("Creating VGG16 for style training")
    style_gen_model = Vgg16()
    if cfg.USE_GPU:
        style_gen_model.to(device)
elif cfg.NET_CROSS_STYLE_LOSS:
    style_gen_model = GeneratorNet(output_full_image=True, extract_features=True)
    if cfg.USE_GPU:
        style_gen_model.to(device)
else:
    style_gen_model = None


# pretrained model loading
if cfg.APPLY_GAUSSIAN_WEIGHT_INIT:
    gen_model.apply(weights_init)
    disc_model.apply(weights_init)

if cfg.ENABLE_PRETRAINED_MODEL_LOAD:
    print("Loading pretrained model (for enabling transfer learning)")
    if cfg.DATASET_SELECT == "photo":
        gen_enc_model_file = os.path.join(cfg.PRETRAINED_MODEL_PATH, "CentralRegion_64_gen_encoder_weights.pt")
        #gen_dec_model_file = os.path.join(cfg.PRETRAINED_MODEL_PATH, "CentralRegion_64_gen_decoder_weights.pt")
        #disc_model_file = os.path.join(cfg.PRETRAINED_MODEL_PATH, "CentralRegion_64_disc_weights.pt")
        #gen_model.load_pretrained_encoder(gen_enc_model_file)
        #gen_model.load_pretrained_decoder(gen_dec_model_file)
        #disc_model.load_model(disc_model_file)
        train_with_style_loss = False
    else:
        if cfg.MASKING_METHOD == "CentralRegion":
            if cfg.MASK_SIZE == 64:
                gen_enc_model_file = os.path.join(cfg.BASE_PROJECT_PATH + 'models/photo/good_model_central_region_64', "CentralRegion_64_gen_encoder_weights.pt")
                gen_model.load_pretrained_encoder(gen_enc_model_file)
                gen_dec_model_file = os.path.join(cfg.BASE_PROJECT_PATH + 'models/photo/good_model_central_region_64', "CentralRegion_64_gen_decoder_weights.pt")
                gen_model.load_pretrained_decoder(gen_dec_model_file)
                #disc_model_file = os.path.join(cfg.PRETRAINED_MODEL_PATH, "RandomRegion_disc_weights.pt")
                #disc_model.load_state_dict(torch.load(disc_model_file))
        else:
            gen_enc_model_file = os.path.join(cfg.PRETRAINED_MODEL_PATH, "RandomRegion_gen_encoder_weights.pt")
            gen_model.load_pretrained_encoder(gen_enc_model_file)
            gen_dec_model_file = os.path.join(cfg.PRETRAINED_MODEL_PATH, "RandomRegion_gen_decoder_weights.pt")
            gen_model.load_pretrained_decoder(gen_dec_model_file)
            #disc_model_file = os.path.join(cfg.PRETRAINED_MODEL_PATH, "RandomRegion_disc_weights.pt")
            #disc_model.load_state_dict(torch.load(disc_model_file))
        train_with_style_loss = True
        if not cfg.EXTERNAL_REF_NET_CROSS_STYLE_LOSS:
            if cfg.NET_CROSS_STYLE_LOSS:
                style_gen_model.load_pretrained_encoder(gen_enc_model_file)
                style_gen_model.load_pretrained_decoder(gen_dec_model_file)
else:
    train_with_style_loss = False

print("Doing arrangements to run & log model...")
if cfg.USE_GPU:
    gen_model.to(device)
    disc_model.to(device)

if cfg.WEIGHT_DECAY:
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=cfg.GEN_LR, weight_decay=cfg.WEIGHT_DECAY_VAL)
else:
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=cfg.GEN_LR) #betas=())
disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=cfg.DISC_LR, betas=(cfg.DISC_BETA1, cfg.DISC_BETA2))


rec_criterion = nn.MSELoss()
adv_criterion = nn.BCEWithLogitsLoss()
style_criterion = StyleLoss()

params = [cfg.BATCH_SIZE,
cfg.NUM_EPOCHS,
cfg.GEN_LR,
cfg.DISC_LR,
cfg.DISC_BETA1,
cfg.DISC_BETA2,
cfg.LAMBDA_REC,
cfg.LAMBDA_ADV]

params_str = '_'.join([str(p) for p in params])
now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
experiment_name = 'logs/' + cfg.DATASET_SELECT + '/experiment_' + date_time + 'masking' + cfg.MASKING_METHOD + '_' + params_str

if cfg.ENABLE_TENSORBOARD:
    print("Setting up experiment logging on Tensorboard")
    writer = SummaryWriter(experiment_name) 
else:
    writer = None

# model_path = 'weights/Morph2Diff/unified/iter/' + experiment_name + "_" + "unfreezecnnon5_transforms" #Diff_RangerLars_lr_1e3_4096_epochs_60_batch_32_vgg16_warmup_10k_cosine_bin_1_2'
# if not os.path.exists(model_path):
#     os.makedirs(model_path)

print("Runing training...")
gen_model, disc_model = train_model(gen_model, 
                disc_model, 
                gen_optimizer,
                disc_optimizer,
                rec_criterion,
                adv_criterion,
                style_criterion,
                train_with_style_loss,
                cfg.LAMBDA_REC,
                cfg.LAMBDA_ADV,
                cfg.LAMBDA_STYLE,
                data_loaders, 
                dataset_sizes,
                cfg.NUM_EPOCHS,
                device, 
                writer,
                style_gen_model)

print("Saving model")

if cfg.MASKING_METHOD == "CentralRegion":
    save_prefix = "CentralRegion_" + str(cfg.MASK_SIZE)
else:
    save_prefix = cfg.MASKING_METHOD

# save model
if cfg.ENABLE_MODEL_SAVE:
    gen_full_model_file = os.path.join(cfg.MODEL_SAVE_PATH, save_prefix + "_gen_full_weights.pt")
    torch.save(gen_model, gen_full_model_file)
    gen_enc_model_file = os.path.join(cfg.MODEL_SAVE_PATH, save_prefix + "_gen_encoder_weights.pt")
    torch.save(gen_model.get_encoder().state_dict(), gen_enc_model_file)
    gen_dec_model_file = os.path.join(cfg.MODEL_SAVE_PATH, save_prefix + "_gen_decoder_weights.pt")
    torch.save(gen_model.get_decoder().state_dict(), gen_dec_model_file)
    disc_model_file = os.path.join(cfg.MODEL_SAVE_PATH, save_prefix + "_disc_weights.pt")
    torch.save(disc_model.state_dict(), disc_model_file)






