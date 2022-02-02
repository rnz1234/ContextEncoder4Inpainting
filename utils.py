import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def evaluate_on_image(masked_image, orig_image, gen_model):
    with torch.no_grad():
        x = masked_image
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])

        masked_image = x.cpu().numpy()[0]
        masked_image_to_show = np.moveaxis(masked_image, 0, -1)
        
        plt.imshow(masked_image_to_show)
        plt.show()

        g_out = gen_model(x)
        out_image = g_out.cpu().numpy()[0]
        out_image_to_show = np.moveaxis(out_image, 0, -1)

        out_image_to_show = build_inpainted_image_center(masked_image_to_show, out_image_to_show, 256, 128)
        
        plt.imshow(out_image_to_show)
        plt.show()

        x = orig_image
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
        in_image = x.cpu().numpy()[0]
        in_image_to_show = np.moveaxis(in_image, 0, -1)
        
        plt.imshow(in_image_to_show)#, masked_image_to_show, out_image_to_show)
        plt.show()


def build_inpainted_image_center(masked_image, reconstructed, image_dim_size, mask_dim_size):
    mask_low_idx = (image_dim_size - mask_dim_size) // 2
    mask_high_idx = mask_low_idx + mask_dim_size
        
    inpainted_image = deepcopy(masked_image)
    inpainted_image[mask_low_idx:mask_high_idx, mask_low_idx:mask_high_idx, :] = reconstructed

    return inpainted_image

