from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import config as cfg
from dataset import *
from train import *
from model import *


if cfg.USE_GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

print(device)


train_set = ImagesDataset(images_dir_path=cfg.DATASET_PATH, 
                set_type=SetType.TrainSet, 
                masking_method=MaskingMethod.CentralRegion, 
                image_dim_size=256, 
                mask_dim_size=128,
                transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

valid_set = ImagesDataset(images_dir_path=cfg.DATASET_PATH, 
                set_type=SetType.ValidSet, 
                masking_method=MaskingMethod.CentralRegion, 
                image_dim_size=256, 
                mask_dim_size=128,
                transform=transforms.Compose([
                transforms.ToTensor(),
                ]))
                

dataset_sizes = {
    'train' : len(train_set),
    'valid' : len(valid_set)
}
   
data_loaders = {
    'train': DataLoader(train_set, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=True, drop_last=True),
    'val': DataLoader(valid_set, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_OF_WORKERS_DATALOADER, pin_memory=True, shuffle=False, drop_last=True)
}


gen_model = GeneratorNet()
disc_model = DiscriminatorNet()

if cfg.USE_GPU:
    gen_model.to(device)
    disc_model.to(device)

gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=cfg.GEN_LR) #betas=())
disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=cfg.DISC_LR, betas=(cfg.DISC_BETA1, cfg.DISC_BETA2))

rec_criterion = nn.MSELoss()
adv_criterion = nn.BCEWithLogitsLoss()

train_model(gen_model, 
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
                device)
