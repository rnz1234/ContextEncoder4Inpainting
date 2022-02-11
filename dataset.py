import glob
from random import randrange
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageChops
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
    def __init__(self, images_dir_path, set_type=SetType.TrainSet, masking_method=MaskingMethod.CentralRegion,
                 image_dim_size=256, mask_dim_size=128, mask_max_pixels=1000, transform=None):
        self.set_type = set_type
        self.image_files = glob.glob(images_dir_path + "/*.jpg")
        if self.set_type == SetType.TrainSet:
            self.set_files = self.image_files[:int(cfg.TRAIN_SET_RATIO * len(self.image_files))]
        else:
            self.set_files = self.image_files[int((1-cfg.VALID_SET_RATIO) * len(self.image_files)):]
        self.random_region_masks_files = glob.glob(cfg.RANDOM_REGION_TEMPLATES_PATH + "/*.png")
        
        self.masking_method = masking_method
        self.image_dim_size = image_dim_size
        self.mask_dim_size = mask_dim_size
        self.mask_max_pixels = mask_max_pixels
        self.transform = transform

    def _mask_central_region(self, image):
        mask_low_idx = (self.image_dim_size - self.mask_dim_size) // 2
        mask_high_idx = mask_low_idx + self.mask_dim_size
        return self._mask_block(image, mask_low_idx, mask_high_idx, mask_low_idx, mask_high_idx)

    def _mask_block(self, image, mask_low_idx, mask_high_idx, mask_low_idy, mask_high_idy):
        masked_image = deepcopy(image)
        # the original parts
        orig_part = deepcopy(image[:, mask_low_idx:mask_high_idx, mask_low_idy:mask_high_idy])
        # mask by putting max pixel value
        masked_image[:, mask_low_idx:mask_high_idx, mask_low_idy:mask_high_idy] = 1
        return masked_image, orig_part

    def _mask_random_block(self, image):
        # TODO : add logic
        number_of_blocks = randrange(0, cfg.MAX_BLOCKS)
        orig_parts = []
        ids = []
        masked_image = deepcopy(image)
        for i in range(number_of_blocks):
            mask_low_idx = randrange(0, self.image_dim_size - self.mask_dim_size)
            mask_low_idy = randrange(0, self.image_dim_size - self.mask_dim_size)
            mask_high_idx = mask_low_idx + self.mask_dim_size
            mask_high_idy = mask_low_idy + self.mask_dim_size
            _, orig_part = self._mask_block(image, mask_low_idx, mask_high_idx, mask_low_idy, mask_high_idy)
            orig_parts.append(orig_part)
            ids.append((mask_low_idx, mask_high_idx, mask_low_idy, mask_high_idy))
        for mask_low_idx, mask_high_idx, mask_low_idy, mask_high_idy in ids:
            masked_image[:, mask_low_idx:mask_high_idx, mask_low_idy:mask_high_idy] = 1
        orig_parts = ImageChops.subtract(image, masked_image)

        return masked_image, orig_parts

    def _mask_fully_random_region(self, image):
        orig_parts = deepcopy(image)
        orig_parts[:, :, :] = 0
        curr_idx = randrange(self.mask_dim_size, self.image_dim_size-self.mask_dim_size)
        curr_idy = randrange(self.mask_dim_size, self.image_dim_size-self.mask_dim_size)
        base_idx = curr_idx
        base_idy = curr_idy
        masked_image = deepcopy(image)
        for i in range(self.mask_max_pixels):
            next_idx = randrange(-1, 2)
            next_idy = randrange(-1, 2)
            # if abs(next_idx-base_idx) > self.mask_dim_size/2 or abs(next_idy-base_idy) > self.mask_dim_size/2:
            #     return masked_image, orig_parts

            if 0 <= curr_idx + next_idx < self.image_dim_size and 0 <= curr_idy + next_idy < self.image_dim_size:
                orig_parts[:, curr_idx + next_idx, curr_idy + next_idy] = image[:, curr_idx + next_idx, curr_idy + next_idy]
                masked_image[:, curr_idx + next_idx, curr_idy + next_idy] = 1
                curr_idx += next_idx
                curr_idy += next_idy
        return masked_image, orig_parts #, base_idx, base_idy

    def _mask_random_region(self, image, idx):
        # import pdb
        # pdb.set_trace()
        orig_parts = deepcopy(image)
        orig_parts[:, :, :] = 0
        newsize = (self.image_dim_size, self.image_dim_size)
        mask_im = Image.open(self.random_region_masks_files[idx % len(self.random_region_masks_files)])
        mask_im = mask_im.resize(newsize)
        mask_im_tensor = self.transform(mask_im)
        mask_im_tensor[mask_im_tensor != 0] = 1
        
        masked_image = deepcopy(image)
        mask_im_tensor_2d = mask_im_tensor.view(self.image_dim_size, self.image_dim_size)
        # the original parts
        orig_parts[:, mask_im_tensor_2d == 1] = deepcopy(image[:, mask_im_tensor_2d == 1])
        # mask by putting max pixel value
        masked_image[:, mask_im_tensor_2d == 1] = 1

        # plt.imshow(np.transpose(masked_image.numpy(), (1, 2, 0)))
        # plt.show()

        # plt.imshow(np.transpose(orig_parts.numpy(), (1, 2, 0)))
        # plt.show()

        return masked_image, orig_parts
        
        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.set_files[idx])
        image = self.transform(image)
        base_idx = 0
        base_idy = 0
        if self.masking_method == MaskingMethod.CentralRegion:
            masked_image, orig_parts = self._mask_central_region(image)
            # plt.imshow(np.transpose(masked_image.numpy(), (1, 2, 0)))
            # plt.show()
        elif self.masking_method == MaskingMethod.RandomBlock:
            masked_image, orig_parts = self._mask_random_block(image)
        elif self.masking_method == MaskingMethod.RandomRegion:
            #masked_image, orig_parts, base_idx, base_idy = self._mask_random_region(image)
            masked_image, orig_parts = self._mask_random_region(image, idx)
        else:
            print("invalid masking method")
            exit()

        # import pdb
        # pdb.set_trace()
        return {"orig_image": image, "masked_image": masked_image, "orig_parts": orig_parts} #, "center_x" : base_idx, "center_y" : base_idy}

    def __len__(self):
        return len(self.set_files)
