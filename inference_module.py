import config as cfg
from utils import *
from model import *
from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import torch
from torchvision import transforms
import random


"""
Inference function - paramters:
input_image_path - input image path
input_mask_path - input mask path
output_image_path - path to store the output image. If None - not saved.
model - either "photo" or "monet". Accordingly to this the correct trained model will be loaded. Default is "photo".
force_mask_on_input - True by default. Ensures that mask is forced on the input image (even if the input image comes already masked).
display_all - display various related images (the masked image, the mask, the reconstruction and the output)
display_output - display the output
pretrained_generator_exact_path - if not None, this is the path for the generator pretraind model - in this case model argument is ignored.
"""
def infer_inpainting(input_image_path, input_mask_path, output_image_path=None, model='photo', force_mask_on_input=True, display_all=False, display_output=False, pretrained_generator_exact_path=None):
    # fix randomness to one used in training
    if cfg.FIXED_RANDOM:
        #print("Fixing random in order to enable reproduction")
        torch.manual_seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)
        random.seed(cfg.RANDOM_SEED)

    # setting device
    if cfg.USE_GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")


    # instantiate model
    if model != "photo" and model != "monet":
        print("unsupported model type")
        return None

    gen_model = GeneratorNet(output_full_image=True) #, extract_features=extract_features)

    # convert to GPU if in use
    if cfg.USE_GPU:
        gen_model.to(device)

    # load model
    if pretrained_generator_exact_path is None:
        if model == "photo":
            pretrained_model_path = cfg.BASE_PROJECT_PATH + 'models/photo/good_model_random_region'
            gen_model_file = os.path.join(pretrained_model_path, "RandomRegion_gen_full_weights.pt")
        elif model == "monet":
            pretrained_model_path = cfg.BASE_PROJECT_PATH + 'models/monet/good_model_random_region'
            gen_model_file = os.path.join(pretrained_model_path, "RandomRegion_gen_full_weights.pt")
    else:
        gen_model_file = pretrained_generator_exact_path
    #gen_model.load_state_dict(torch.load(gen_model_file))
    gen_model = torch.load(gen_model_file)

    # pass to eval model for inference
    gen_model.eval()

    # setting transforms for image (resize, to tensor, norm)
    # and one also without normalization in order to use it for display / the actual integration with inpainted parts later
    transforms_list = []
    transforms_list_no_norm = []
    if cfg.TO_RESIZE:
        #print("resizing images to", cfg.RESIZE_DIM)
        transforms_list.append(transforms.Resize((cfg.RESIZE_DIM, cfg.RESIZE_DIM)))
        transforms_list_no_norm.append(transforms.Resize((cfg.RESIZE_DIM, cfg.RESIZE_DIM)))
    transforms_list.append(transforms.ToTensor())
    transforms_list_no_norm.append(transforms.ToTensor())
    if cfg.TO_NORMALIZE:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))#(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])) #(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transforms_list = transforms.Compose(transforms_list)
    transforms_list_no_norm = transforms.Compose(transforms_list_no_norm)

    # apply transforms on input image
    input_image = Image.open(input_image_path)
    input_image_trns = transforms_list(input_image)
    input_image_trns_no_norm = transforms_list_no_norm(input_image)

    # setting corresponding transforms for mask (resize, to tensor)
    mask_im = Image.open(input_mask_path)
    transforms_list_mask = []
    if cfg.TO_RESIZE:
        transforms_list_mask.append(transforms.Resize((cfg.RESIZE_DIM, cfg.RESIZE_DIM)))
    transforms_list_mask.append(transforms.ToTensor())

    transforms_list_mask = transforms.Compose(transforms_list_mask)

    # apply transforms on mask image
    mask_im_tensor = transforms_list_mask(mask_im)
    
    if display_all:
        # display mask
        plt.imshow(np.transpose(mask_im_tensor.numpy(), (1, 2, 0)))
        plt.show()

    input_nonorm = np.transpose(input_image_trns_no_norm.numpy(), (1, 2, 0))
    if display_all:
        # display input image
        plt.imshow(input_nonorm)
        plt.show()
    
    
    # generate masked image using input and mask images (optional - to enforce mask on image in case of small incosistencies)
    masked_image = deepcopy(input_image_trns)
   
    if force_mask_on_input:
        # mask by putting max pixel value
        masked_image[mask_im_tensor != 0] = 1

    masked_image = masked_image.view(1, 3, cfg.RESIZE_DIM, cfg.RESIZE_DIM)
    

    # apply model on masked image
    with torch.no_grad():
        reconstructed = gen_model(masked_image.to(device))[0]

    # import pdb
    # pdb.set_trace()

    # de-normalize reconstructed
    x = reconstructed
    if cfg.TO_NORMALIZE:
        MEAN = torch.tensor([0.5,0.5,0.5])#([0.485, 0.456, 0.406]) #np.array([0.485, 0.456, 0.406]) #
        STD = torch.tensor([0.5,0.5,0.5])#([0.229, 0.224, 0.225]) #np.array([0.229, 0.224, 0.225]) #

        reconstructed_unnorm = x.cpu() * STD[:, None, None] + MEAN[:, None, None]
    else:
        reconstructed_unnorm = x.cpu()

    #print(x_unnorm.shape)

    if display_all:
        # display reconstructed
        plt.imshow(np.transpose(reconstructed_unnorm.numpy(), (1, 2, 0)))
        plt.show()

    # apply the inpainted (reconstructed parts) to the masked image to create the output image
    output_image_unnorm = deepcopy(input_image_trns_no_norm)
    if pretrained_generator_exact_path is None:
        output_image_unnorm[mask_im_tensor != 0] = reconstructed_unnorm[mask_im_tensor != 0]
    elif "CentralRegion_64" in pretrained_generator_exact_path:
        # optimization for this special case - to indicate the capabilities of a mask-oriented model
        mask_low_idx = (128 - 64) // 2
        mask_high_idx = mask_low_idx + 64
            
        output_image_unnorm[:, mask_low_idx:mask_high_idx, mask_low_idx:mask_high_idx] = reconstructed_unnorm
    else:
        output_image_unnorm[mask_im_tensor != 0] = reconstructed_unnorm[mask_im_tensor != 0]

    output = np.transpose(output_image_unnorm.numpy(), (1, 2, 0))
    if display_all or display_output:
        f, axarr = plt.subplots(1, 2, figsize=(16,16))
        axarr[0].imshow(input_nonorm)
        axarr[1].imshow(output)
        # plt.imshow(output)
        plt.show()

    if output_image_path is not None:
        plt.imsave(output_image_path, output)
        plt.show()

    return output



# infer_inpainting(input_image_path=cfg.BASE_PROJECT_PATH+'data/my_examples/validation_example.jpg',
#                  input_mask_path=cfg.BASE_PROJECT_PATH+'data/my_examples/validation_example_mask.jpg', 
#                 output_image_path='', 
#                 model='photo')

# infer_inpainting(input_image_path=cfg.BASE_PROJECT_PATH+'data/examples/example1.jpg',
#                  input_mask_path=cfg.BASE_PROJECT_PATH+'data/examples/example1_mask.jpg', 
#                 output_image_path='', 
#                 model='photo',
#                 display=True)

# infer_inpainting(input_image_path=cfg.BASE_PROJECT_PATH+'data/examples/example2.jpg',
#                  input_mask_path=cfg.BASE_PROJECT_PATH+'data/examples/example2_mask.jpg', 
#                 output_image_path=cfg.BASE_PROJECT_PATH+'data/examples/my_outputs/example_output.png', 
#                 model='monet',
#                 display_all=False,
#                 display_output=True)

# infer_inpainting(input_image_path=cfg.BASE_PROJECT_PATH+'data/my_examples/000ded5c41.jpg',
#                  input_mask_path=cfg.BASE_PROJECT_PATH+'data/my_examples/000ded5c41_mask.jpg', 
#                 output_image_path='', 
#                 model='photo',
#                 display=True)
    

