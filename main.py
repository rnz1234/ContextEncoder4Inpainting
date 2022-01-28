from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import config as cfg
from dataset import *
from train import *



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


model = None

train_model(model, data_loaders, dataset_sizes)
