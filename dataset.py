import glob
import torch
from torch.utils.data import Dataset 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import config as cfg


class MaskingMethod:
    CentralRegion = 0
    RandomBlock = 1
    RandomRegion = 2

class SetType:
    TrainSet = 0
    ValidSet = 1
    


class ImagesDataset(Dataset):
    def __init__(self, images_dir_path, set_type=SetType.TrainSet, masking_method=MaskingMethod.CentralRegion, image_dim_size=256, mask_dim_size=128, transform=None):
        self.set_type = set_type
        self.image_files = glob.glob(images_dir_path + "/*.jpg")
        if self.set_type == SetType.TrainSet:
            self.set_files = self.image_files[:int(cfg.TRAIN_SET_RATIO*len(self.image_files))]
        else:
            self.set_files = self.image_files[int(cfg.TRAIN_SET_RATIO*len(self.image_files)):]
        self.masking_method = masking_method
        self.image_dim_size = image_dim_size
        self.mask_dim_size = mask_dim_size
        self.transform = transform

    def _mask_central_region(self, image):
        mask_low_idx = (self.image_dim_size - self.mask_dim_size) // 2
        mask_high_idx = mask_low_idx + self.mask_dim_size
        masked_image = deepcopy(image)
        # the original parts
        orig_parts = deepcopy(image[:, mask_low_idx:mask_high_idx, mask_low_idx:mask_high_idx])
        # mask by putting max pixel value
        masked_image[:, mask_low_idx:mask_high_idx, mask_low_idx:mask_high_idx] = 1
        #import pdb
        #pdb.set_trace()
        #img = plt.imshow(np.transpose(masked_image.numpy(), (1, 2, 0)))
        #plt.show()
        return masked_image, orig_parts

    def _mask_random_block(self, image):
        # TODO : add logic
        orig_parts = None
        masked_image = deepcopy(image)
        return masked_image, orig_parts

    def _mask_random_region(self, image):
        # TODO : add logic
        orig_parts = None
        masked_image = deepcopy(image)
        return masked_image, orig_parts

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.set_files[idx])
        image = self.transform(image)
        if self.masking_method == MaskingMethod.CentralRegion:
            masked_image, orig_parts = self._mask_central_region(image)
        elif self.masking_method == MaskingMethod.RandomBlock:
            masked_image, orig_parts = self._mask_random_block(image)
        elif self.masking_method == MaskingMethod.RandomRegion:
            masked_image, orig_parts = self._mask_random_region(image)
        else:
            print("invalid masking method")
            exit()

        # import pdb
        # pdb.set_trace()
        return {"orig_image" : image, "masked_image" : masked_image, "orig_parts" : orig_parts}

    def __len__(self):
        return len(self.set_files)