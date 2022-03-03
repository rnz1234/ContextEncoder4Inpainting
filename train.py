from tqdm import tqdm
import time 
import torch
import copy
import numpy as np
from utils import *


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
        running_dloss_fake = 0.0
        running_dloss_real = 0.0

        #validate(gen_model, disc_model, rec_criterion, adv_criterion, lambda_rec, lambda_adv, data_loaders['valid'], dataset_sizes['valid'], epoch, device, writer)
        

        for i, batch in enumerate(tqdm(data_loaders['train'])):
            # import pdb
            # pdb.set_trace()
            orig_image = batch['orig_image'].to(device)
            real_parts = batch['orig_parts'].to(device)
            masked_image = batch['masked_image'].to(device)

            if (epoch+1) % cfg.NUM_EPOCHS_PER_DISPLAY == 0:
                if i == 0:
                    print("RESULTS ON TRAIN SET:")
                    # import pdb
                    # pdb.set_trace()
                    #masked_image[0].view(1, masked_image[0].shape[0], masked_image[0].shape[1], masked_image[0].shape[2])
                    for j in range(8):
                        evaluate_on_image(masked_image[j], orig_image[j], real_parts[j], gen_model, sum_for_random=True)
                

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Training generator
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #if epoch % 5 == 0:
            train_gen = True
            if cfg.DOWNSCALE_GEN_TRAIN:
                if i % cfg.DOWNSCALE_GEN_TRAIN_RATIO == 0:
                    train_gen = True
                else:
                    train_gen = False

            if train_gen:
                # zero gradient
                gen_optimizer.zero_grad()

                # run generator model
                g_out = gen_model(masked_image)

                #print("1")

                # recognition loss
                if cfg.MASKING_METHOD == "CentralRegion":
                    g_rec_loss = rec_criterion(g_out, real_parts)
                else:
                    g_rec_loss = rec_criterion(g_out, orig_image)

                #if epoch > 15:
                # run discriminator model
                d_out = disc_model(g_out)

                #print("2")

                # adversarial loss : try to fool discriminator (make generator generate images discrimintor cannot mark as fake)
                g_adv_loss = adv_criterion(d_out, torch.ones_like(d_out))

                # full loss as described in paper
                if cfg.CANCEL_ADV_TRAIN:
                    g_loss = joined_loss(lambda_rec, g_rec_loss, 0, 0)
                else:
                    g_loss = joined_loss(lambda_rec, g_rec_loss, lambda_adv, g_adv_loss)

                running_gadv_loss += g_adv_loss.item() * orig_image.size(0)
                #else:
                    #g_loss = g_rec_loss

                # backprop
                g_loss.backward()
                gen_optimizer.step()

                #print("3")

            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Training discriminator
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            #if epoch > 15:
            if not cfg.CANCEL_ADV_TRAIN:
                # zero gradient
                disc_optimizer.zero_grad()

                # run discriminator to train identification of real data
                if cfg.MASKING_METHOD == "CentralRegion":
                    d_out_real = disc_model(real_parts)
                else:
                    d_out_real = disc_model(orig_image)
                #d_out_real = disc_model(orig_image)
                d_out_real_loss = adv_criterion(d_out_real, torch.ones_like(d_out))

                # run discriminator to train identification of fake data
                d_out_fake = disc_model(g_out.detach())
                d_out_fake_loss = adv_criterion(d_out_fake, torch.zeros_like(d_out_fake))

                # full adversarial loss 
                d_loss = (d_out_real_loss + d_out_fake_loss) / 2

                # backprop
                d_loss.backward()
                disc_optimizer.step()

                running_dloss += d_loss.item() * orig_image.size(0)
                running_dloss_fake += d_out_fake_loss.item() *  orig_image.size(0)
                running_dloss_real += d_out_real_loss.item() *  orig_image.size(0)

            running_gloss += g_loss.item() * orig_image.size(0)
            running_grec_loss += g_rec_loss.item() * orig_image.size(0)
            
        # calculate epoch losses
        epoch_dloss = running_dloss / dataset_sizes['train']
        epoch_gloss = running_gloss / dataset_sizes['train']
        epoch_grec_loss = running_grec_loss / dataset_sizes['train']
        epoch_gadv_loss = running_gadv_loss / dataset_sizes['train']
        epoch_dloss_fake = running_dloss_fake / dataset_sizes['train']
        epoch_dloss_real = running_dloss_real / dataset_sizes['train']

        
        print("Epoch {epoch}".format(epoch=epoch))
        print("Training | Disc Loss: {disc_loss}, | Gen Loss: {gen_loss}, | gRec Loss: {rec_loss}, | gAdv Loss: {gadv_loss}, |".format(disc_loss=epoch_dloss, gen_loss=epoch_gloss, rec_loss=epoch_grec_loss, gadv_loss=epoch_gadv_loss))

        if cfg.ENABLE_TENSORBOARD:
            writer.add_scalar('Discriminator Fake Loss/{}'.format('train'), epoch_dloss_fake, epoch)
            writer.add_scalar('Discriminator Real Loss/{}'.format('train'), epoch_dloss_real, epoch)
            writer.add_scalar('Discriminator Loss/{}'.format('train'), epoch_dloss, epoch)
            writer.add_scalar('Generator Loss/{}'.format('train'), epoch_gloss, epoch)
            writer.add_scalar('Generator Rec. Loss/{}'.format('train'), epoch_grec_loss, epoch)
            writer.add_scalar('Generator Adv. Loss/{}'.format('train'), epoch_gadv_loss, epoch)
            

        # run validation
        if (epoch+1) % cfg.NUM_EPOCHS_PER_DISPLAY == 0:
            show_examples = True
        else:
            show_examples = False
        validate(gen_model, disc_model, rec_criterion, adv_criterion, lambda_rec, lambda_adv, data_loaders['valid'], dataset_sizes['valid'], epoch, device, writer, show_examples)
                
        

    return gen_model, disc_model




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
                writer,
                show_examples):

    gen_model.eval()
    disc_model.eval()

    running_dloss = 0.0
    running_gloss = 0.0
    running_grec_loss = 0.0
    running_gadv_loss = 0.0
    running_dloss_fake = 0.0
    running_dloss_real = 0.0
    running_prob_correct = 0.0
    running_prob_real_correct = 0.0
    running_prob_fake_correct = 0.0

    for i, batch in enumerate(tqdm(data_loader_valid)):
        orig_image = batch['orig_image'].to(device)
        real_parts = batch['orig_parts'].to(device)
        masked_image = batch['masked_image'].to(device)

        
        # avoid calculating gradients
        with torch.no_grad():
            if cfg.SHOW_EXAMPLES_RESULTS_ON_VALID_SET:
                if show_examples:
                    if i == 0:
                        print("RESULTS ON VALID SET:")
                        # import pdb
                        # pdb.set_trace()
                        #masked_image[0].view(1, masked_image[0].shape[0], masked_image[0].shape[1], masked_image[0].shape[2])
                        for j in range(16):
                            evaluate_on_image(masked_image[j], orig_image[j], real_parts[j], gen_model, sum_for_random=True)
                    

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
            if cfg.MASKING_METHOD == "CentralRegion":
                d_out_real = disc_model(real_parts)
            else:
                d_out_real = disc_model(orig_image)

            d_out_real_prob = torch.nn.Sigmoid()(d_out_real)
            #d_out_real_prob = d_out_real_prob.view(cfg.BATCH_SIZE,-1).mean(1, keepdim=True)
            d_out_real_prob = torch.mean(d_out_real_prob)
            
            d_out_real_loss = adv_criterion(d_out_real, torch.ones_like(d_out))

            # run discriminator to train identification of fake data
            d_out_fake = disc_model(g_out.detach())
            d_out_fake_loss = adv_criterion(d_out_fake, torch.zeros_like(d_out_fake))

            d_out_fake_prob = 1-torch.nn.Sigmoid()(d_out_fake)
            d_out_fake_prob = torch.mean(d_out_fake_prob)

            d_out_prob_correct = (d_out_real_prob + d_out_fake_prob) / 2

            # full adversarial loss 
            d_loss = (d_out_real_loss + d_out_fake_loss) / 2
            

        running_dloss += d_loss.item() * orig_image.size(0)
        running_gloss += g_loss.item() * orig_image.size(0)
        running_grec_loss += g_rec_loss.item() * orig_image.size(0)
        running_gadv_loss += g_adv_loss.item() * orig_image.size(0)
        running_dloss_fake += d_out_fake_loss.item() *  orig_image.size(0)
        running_dloss_real += d_out_real_loss.item() *  orig_image.size(0)
        running_prob_correct += d_out_prob_correct.item() * orig_image.size(0)
        running_prob_real_correct += d_out_real_prob.item() * orig_image.size(0)
        running_prob_fake_correct += d_out_fake_prob.item() * orig_image.size(0)
            
    # calculate epoch losses
    epoch_dloss = running_dloss / dataset_size_valid
    epoch_gloss = running_gloss / dataset_size_valid
    epoch_grec_loss = running_grec_loss / dataset_size_valid
    epoch_gadv_loss = running_gadv_loss / dataset_size_valid
    epoch_dloss_fake = running_dloss_fake / dataset_size_valid
    epoch_dloss_real = running_dloss_real / dataset_size_valid
    epoch_prob_correct = running_prob_correct / dataset_size_valid
    epoch_prob_real_correct = running_prob_real_correct / dataset_size_valid
    epoch_prob_fake_correct = running_prob_fake_correct / dataset_size_valid

    print("Validation | Disc Loss: {disc_loss}, | Gen Loss: {gen_loss}, | gRec Loss: {rec_loss}, | gAdv Loss: {gadv_loss}, |".format(disc_loss=epoch_dloss, gen_loss=epoch_gloss, rec_loss=epoch_grec_loss, gadv_loss=epoch_gadv_loss))

    if cfg.ENABLE_TENSORBOARD:
        writer.add_scalar('Discriminator Fake Loss/{}'.format('train'), epoch_dloss_fake, epoch)
        writer.add_scalar('Discriminator Real Loss/{}'.format('train'), epoch_dloss_real, epoch)
        writer.add_scalar('Discriminator Loss/{}'.format('valid'), epoch_dloss, epoch)
        writer.add_scalar('Generator Loss/{}'.format('valid'), epoch_gloss, epoch)
        writer.add_scalar('Generator Rec. Loss/{}'.format('valid'), epoch_grec_loss, epoch)
        writer.add_scalar('Generator Adv. Loss/{}'.format('valid'), epoch_gadv_loss, epoch)
        writer.add_scalar('Discriminator Fake Correct Prob/{}'.format('valid'), epoch_prob_fake_correct, epoch)
        writer.add_scalar('Discriminator Real Correct Prob/{}'.format('valid'), epoch_prob_real_correct, epoch)
        writer.add_scalar('Discriminator Total Correct Prob/{}'.format('valid'), epoch_prob_correct, epoch)



