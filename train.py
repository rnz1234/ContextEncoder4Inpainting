from tqdm import tqdm
import time 
import torch
import copy
import numpy as np
from utils import *


def joined_loss(lambda_rec, g_rec_loss, lambda_adv, g_adv_loss):
    return lambda_rec*g_rec_loss + lambda_adv*g_adv_loss


def train_model(gen_model, 
                disc_model, 
                gen_optimizer,
                disc_optimizer,
                rec_criterion,
                adv_criterion,
                lambda_rec,
                lambda_adv,
                data_loaders, 
                dataset_sizes,
                num_epochs,
                device,
                writer):
    for epoch in range(num_epochs):

        gen_model.train()
        disc_model.train()

        running_dloss = 0.0
        running_gloss = 0.0
        running_grec_loss = 0.0
        running_gadv_loss = 0.0

        #validate(gen_model, disc_model, rec_criterion, adv_criterion, lambda_rec, lambda_adv, data_loaders['valid'], dataset_sizes['valid'], epoch, device, writer)
        

        for batch in tqdm(data_loaders['train']):
            # import pdb
            # pdb.set_trace()
            orig_image = batch['orig_image'].to(device)
            real_parts = batch['orig_parts'].to(device)
            masked_image = batch['masked_image'].to(device)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Training generator
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # zero gradient
            gen_optimizer.zero_grad()

            # run generator model
            g_out = gen_model(masked_image)

            # recognition loss
            g_rec_loss = rec_criterion(g_out, real_parts)

            # run discriminator model
            d_out = disc_model(g_out)

            # adversarial loss : try to fool discriminator (make generator generate images discrimintor cannot mark as fake)
            g_adv_loss = adv_criterion(d_out, torch.ones_like(d_out))

            # full loss as described in paper
            g_loss = joined_loss(lambda_rec, g_rec_loss, lambda_adv, g_adv_loss)

            # backprop
            g_loss.backward()
            gen_optimizer.step()

            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Training discriminator
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # zero gradient
            disc_optimizer.zero_grad()

            # run discriminator to train identification of real data
            d_out_real = disc_model(real_parts)
            d_adv_real_loss = adv_criterion(d_out_real, torch.ones_like(d_out))

            # run discriminator to train identification of fake data
            d_out_fake = disc_model(g_out.detach())
            d_out_fake_loss = adv_criterion(d_out_fake, torch.zeros_like(d_out_fake))

            # full adversarial loss 
            d_loss = (d_adv_real_loss + d_out_fake_loss) / 2

            # backprop
            d_loss.backward()
            disc_optimizer.step()

            running_dloss += d_loss.item() * orig_image.size(0)
            running_gloss += g_loss.item() * orig_image.size(0)
            running_grec_loss += g_rec_loss.item() * orig_image.size(0)
            running_gadv_loss += g_adv_loss.item() * orig_image.size(0)
            
        # calculate epoch losses
        epoch_dloss = running_dloss / dataset_sizes['train']
        epoch_gloss = running_gloss / dataset_sizes['train']
        epoch_grec_loss = running_grec_loss / dataset_sizes['train']
        epoch_gadv_loss = running_gadv_loss / dataset_sizes['train']

        
        print("Epoch {epoch}".format(epoch=epoch))
        print("Training | Disc Loss: {disc_loss}, | Gen Loss: {gen_loss}, | gRec Loss: {rec_loss}, | gAdv Loss: {gadv_loss}, |".format(disc_loss=epoch_dloss, gen_loss=epoch_gloss, rec_loss=epoch_grec_loss, gadv_loss=epoch_gadv_loss))

        writer.add_scalar('Discriminator Loss/{}'.format('train'), epoch_dloss, epoch)
        writer.add_scalar('Generator Loss/{}'.format('train'), epoch_gloss, epoch)
        writer.add_scalar('Generator Rec. Loss/{}'.format('train'), epoch_grec_loss, epoch)
        writer.add_scalar('Generator Adv. Loss/{}'.format('train'), epoch_gadv_loss, epoch)

        # run validation
        validate(gen_model, disc_model, rec_criterion, adv_criterion, lambda_rec, lambda_adv, data_loaders['valid'], dataset_sizes['valid'], epoch, device, writer)
                
    return gen_model




def validate(gen_model, 
                disc_model, 
                rec_criterion,
                adv_criterion,
                lambda_rec,
                lambda_adv,
                data_loader_valid, 
                dataset_size_valid,
                epoch,
                device,
                writer):

    gen_model.eval()
    disc_model.eval()

    running_dloss = 0.0
    running_gloss = 0.0
    running_grec_loss = 0.0
    running_gadv_loss = 0.0

    for i, batch in enumerate(data_loader_valid):
        orig_image = batch['orig_image'].to(device)
        real_parts = batch['orig_parts'].to(device)
        masked_image = batch['masked_image'].to(device)

        
        # avoid calculating gradients
        with torch.no_grad():
            if i == 0:
                # import pdb
                # pdb.set_trace()
                #masked_image[0].view(1, masked_image[0].shape[0], masked_image[0].shape[1], masked_image[0].shape[2])
                evaluate_on_image(masked_image[0], orig_image[0], gen_model)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calculating generator loss
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # run generator model
            g_out = gen_model(masked_image)

            # recognition loss
            g_rec_loss = rec_criterion(g_out, real_parts)

             # run discriminator model
            d_out = disc_model(g_out)

            # adversarial loss : try to fool discriminator (make generator generate images discrimintor cannot mark as fake)
            g_adv_loss = adv_criterion(d_out, torch.ones_like(d_out))

            # full loss as described in paper
            g_loss = joined_loss(lambda_rec, g_rec_loss, lambda_adv, g_adv_loss)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Calculating discriminator loss
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # run discriminator to train identification of real data
            d_out_real = disc_model(real_parts)
            d_adv_real_loss = adv_criterion(d_out_real, torch.ones_like(d_out))

            # run discriminator to train identification of fake data
            d_out_fake = disc_model(g_out.detach())
            d_out_fake_loss = adv_criterion(d_out_fake, torch.zeros_like(d_out_fake))

            # full adversarial loss 
            d_loss = (d_adv_real_loss + d_out_fake_loss) / 2

        running_dloss += d_loss.item() * orig_image.size(0)
        running_gloss += g_loss.item() * orig_image.size(0)
        running_grec_loss += g_rec_loss.item() * orig_image.size(0)
        running_gadv_loss += g_adv_loss.item() * orig_image.size(0)
            
    # calculate epoch losses
    epoch_dloss = running_dloss / dataset_size_valid
    epoch_gloss = running_gloss / dataset_size_valid
    epoch_grec_loss = running_grec_loss / dataset_size_valid
    epoch_gadv_loss = running_gadv_loss / dataset_size_valid

    print("Validation | Disc Loss: {disc_loss}, | Gen Loss: {gen_loss}, | gRec Loss: {rec_loss}, | gAdv Loss: {gadv_loss}, |".format(disc_loss=epoch_dloss, gen_loss=epoch_gloss, rec_loss=epoch_grec_loss, gadv_loss=epoch_gadv_loss))

    writer.add_scalar('Discriminator Loss/{}'.format('valid'), epoch_dloss, epoch)
    writer.add_scalar('Generator Loss/{}'.format('valid'), epoch_gloss, epoch)
    writer.add_scalar('Generator Rec. Loss/{}'.format('valid'), epoch_grec_loss, epoch)
    writer.add_scalar('Generator Adv. Loss/{}'.format('valid'), epoch_gadv_loss, epoch)



