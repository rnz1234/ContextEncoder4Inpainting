from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
import os
import random

import config as cfg
from dataset import *
from train import *
from model import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

if cfg.FIXED_RANDOM:
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)


if cfg.USE_GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

print(device)


train_set = ImagesDataset(images_dir_path=cfg.DATASET_PATH, 
                set_type=SetType.TrainSet, 
                masking_method=eval("MaskingMethod."+cfg.MASKING_METHOD), 
                image_dim_size=cfg.IMAGE_SIZE, 
                mask_dim_size=cfg.MASK_SIZE,
                transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

if cfg.SHOW_IMAGE:
    for image in train_set:
        try:
            data = image
            plt.imshow(np.transpose(data["masked_image"].numpy(), (1, 2, 0)))
            plt.show()
            plt.imshow(np.transpose(data["orig_parts"].numpy(), (1, 2, 0)))
            plt.show()
        except KeyboardInterrupt:
            exit()

valid_set = ImagesDataset(images_dir_path=cfg.DATASET_PATH, 
                set_type=SetType.ValidSet, 
                masking_method=eval("MaskingMethod."+cfg.MASKING_METHOD), 
                image_dim_size=cfg.IMAGE_SIZE, 
                mask_dim_size=cfg.MASK_SIZE,
                transform=transforms.Compose([
                transforms.ToTensor(),
                ]))
                

dataset_sizes = {
    'train' : len(train_set),
    'valid' : len(valid_set)
}
   
data_loaders = {
    'train': DataLoader(train_set, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=True, drop_last=True),
    'valid': DataLoader(valid_set, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=False, drop_last=True)
}

if cfg.MASKING_METHOD == "CentralRegion":
    gen_model = GeneratorNet(output_full_image=False)
else:
    gen_model = GeneratorNet(output_full_image=True)
    
disc_model = DiscriminatorNet()

if cfg.USE_GPU:
    gen_model.to(device)
    disc_model.to(device)

gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=cfg.GEN_LR) #betas=())
disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=cfg.DISC_LR, betas=(cfg.DISC_BETA1, cfg.DISC_BETA2))

rec_criterion = nn.MSELoss()
adv_criterion = nn.BCEWithLogitsLoss()

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

writer = SummaryWriter(experiment_name) 

# model_path = 'weights/Morph2Diff/unified/iter/' + experiment_name + "_" + "unfreezecnnon5_transforms" #Diff_RangerLars_lr_1e3_4096_epochs_60_batch_32_vgg16_warmup_10k_cosine_bin_1_2'
# if not os.path.exists(model_path):
#     os.makedirs(model_path)

gen_model = train_model(gen_model, 
                disc_model, 
                gen_optimizer,
                disc_optimizer,
                rec_criterion,
                adv_criterion,
                cfg.LAMBDA_REC,
                cfg.LAMBDA_ADV,
                data_loaders, 
                dataset_sizes,
                cfg.NUM_EPOCHS,
                device, 
                writer)




