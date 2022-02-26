import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import config as cfg

def evaluate_on_image(masked_image, orig_image, real_parts, gen_model, sum_for_random=False):
    with torch.no_grad():

        f, axarr = plt.subplots(1, 3, figsize=(16,16))

        x = masked_image
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])

        if cfg.TO_NORMALIZE:
            MEAN = torch.tensor([0.5,0.5,0.5])#([0.485, 0.456, 0.406]) #np.array([0.485, 0.456, 0.406]) #
            STD = torch.tensor([0.5,0.5,0.5])#([0.229, 0.224, 0.225]) #np.array([0.229, 0.224, 0.225]) #

            x_unnorm = x.cpu() * STD[:, None, None] + MEAN[:, None, None]
        else:
            x_unnorm = x.cpu()

        masked_image = x_unnorm.numpy()[0] # .cpu()
        masked_image_to_show = np.moveaxis(masked_image, 0, -1)
        
        #plt.imshow(masked_image_to_show)
        
        axarr[0].imshow(masked_image_to_show)
        #plt.show()

        # import pdb
        # pdb.set_trace()

        g_out = gen_model(x)

        if cfg.TO_NORMALIZE:
            g_out = g_out.cpu() * STD[:, None, None] + MEAN[:, None, None]
        else:
            g_out = g_out.cpu()

        out_image = g_out.numpy()[0] #.cpu()
        out_image_to_show = np.moveaxis(out_image, 0, -1)
        
        if cfg.MASKING_METHOD == "CentralRegion":
            out_image_to_show = build_inpainted_image_center(masked_image_to_show, out_image_to_show, cfg.IMAGE_SIZE, cfg.MASK_SIZE)
            #out_image_to_show = build_inpainted_image_full(masked_image_to_show, out_image_to_show, real_parts)


        if sum_for_random:
            if cfg.MASKING_METHOD == "RandomRegion":
                out_image_to_show = build_inpainted_image_full(masked_image_to_show, out_image_to_show, real_parts)

        # import pdb
        # pdb.set_trace()

        #plt.imshow(out_image_to_show)
        
        axarr[1].imshow(out_image_to_show)#, vmin=0, vmax=1)
        #plt.show()

        x = orig_image
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])

        if cfg.TO_NORMALIZE:
            x = x.cpu() * STD[:, None, None] + MEAN[:, None, None]
        else:
            x = x.cpu()
        

        in_image = x.numpy()[0] #.cpu()
        in_image_to_show = np.moveaxis(in_image, 0, -1)
        
          
        #plt.imshow(in_image_to_show)#, masked_image_to_show, out_image_to_show)
        axarr[2].imshow(in_image_to_show)
        plt.show()


def build_inpainted_image_center(masked_image, reconstructed, image_dim_size, mask_dim_size):
    mask_low_idx = (image_dim_size - mask_dim_size) // 2
    mask_high_idx = mask_low_idx + mask_dim_size
        
    inpainted_image = deepcopy(masked_image)
    inpainted_image[mask_low_idx:mask_high_idx, mask_low_idx:mask_high_idx, :] = reconstructed

    return inpainted_image


def get_mask(real_parts):
    mask = deepcopy(real_parts.cpu().numpy()[0])
    mask[mask != 0] = 1
    return(mask)

def build_inpainted_image_full(masked_image, reconstructed, real_parts):
    inpainted_image = deepcopy(masked_image)
    # print(masked_image.shape)
    # print(reconstructed.shape)
    # print(real_parts.shape)
    inpainted_image[get_mask(real_parts) == 1] = reconstructed[get_mask(real_parts) == 1]
    #inpainted_image = deepcopy(get_mask(real_parts))#(reconstructed) #masked_image + 
    

    return inpainted_image #reconstructed #

