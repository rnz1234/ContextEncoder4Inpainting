from tqdm import tqdm
import time 
import torch
import copy



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
                device):
    for epoch in range(num_epochs):

        running_dloss = 0.0
        running_gloss = 0.0
        running_grec_loss = 0.0
        running_gadv_loss = 0.0

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
            g_loss = lambda_rec*g_rec_loss + lambda_adv*g_adv_loss

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

        
        print("Epoch {epoch}".format(epoch))
        print("Training | Disc Loss: {disc_loss}, | Gen Loss: {gen_loss}, | gRec Loss: {rec_loss}, | gAdv Loss: {gadv_loss}, |".format(epoch_dloss, epoch_gloss, epoch_grec_loss, epoch_gadv_loss))





        



    