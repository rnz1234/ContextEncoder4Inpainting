import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import config as cfg
from torchvision import models

class StructureType:
	Compress = 0
	Expand = 1

class ActivationType:
	ReLU = 0
	LeakyReLU = 1

class NormType:
	BatchNorm = 0
	InstanceNorm = 1


# Basic layers structure for the networks
def get_basic_structure(input_size,
						output_size,
						structure_type, 
						activation_type,
						filter_size=4,
						stride_size=2,
						padding=1,
						normalize=True,
						**kwargs): # kwargs includes parameters for sub-modes (like type of norm)
	# Compress structure
	if structure_type == StructureType.Compress:
		# Conv layer
		structure = [nn.Conv2d(input_size, output_size, filter_size, stride_size, padding)]
		
		# Normalization layer
		if normalize:
			eps = kwargs.get('eps', 0.8)
			structure.append(nn.BatchNorm2d(output_size, eps))
		
		# Activation
		if activation_type == ActivationType.LeakyReLU:
			neg_slope = kwargs.get('neg_slope', 0.2)
			structure.append(nn.LeakyReLU(neg_slope))
		elif activation_type == ActivationType.ReLU:
			structure.append(nn.ReLU())
		else:
			print("unsupported activation type, aborting")
			exit()
			
	# Expand structure
	elif structure_type == StructureType.Expand:
		# Conv layer
		structure = [nn.ConvTranspose2d(input_size, output_size, filter_size, stride_size, padding)]
		
		# Normalization layer
		if normalize:
			eps = kwargs.get('eps', 0.8)
			norm_type = kwargs.get('norm', NormType.BatchNorm)
			if norm_type == NormType.BatchNorm:
				structure.append(nn.BatchNorm2d(output_size, eps))
			else:
				structure.append(nn.InstanceNorm2d(output_size))

		# Activation
		if activation_type == ActivationType.LeakyReLU:
			neg_slope = kwargs.get('neg_slope', 0.2)
			structure.append(nn.LeakyReLU(neg_slope))
		elif activation_type == ActivationType.ReLU:
			structure.append(nn.ReLU())
		else:
			print("unsupported activation type, aborting")
			exit()
		
	else:
		print("unsupported structure type, aborting")
		exit()
	
	return structure


# Generator network (context encoder)
class GeneratorNet(nn.Module):
	def __init__(self, output_full_image=False, output_size=128, extract_features=False):
		super(GeneratorNet, self).__init__()

		self.output_full_image = output_full_image
		self.output_size = output_size
		self.extract_features = extract_features
		self.features = []

		# we use a fixed encoder architecture for simplicity
		if self.output_full_image:
			self.base_modules_enc = [*get_basic_structure(3, 64, StructureType.Compress, ActivationType.LeakyReLU),
				*get_basic_structure(64, 64, StructureType.Compress, ActivationType.LeakyReLU),
				*get_basic_structure(64, 128, StructureType.Compress, ActivationType.LeakyReLU),
				*get_basic_structure(128, 256, StructureType.Compress, ActivationType.LeakyReLU),
				*get_basic_structure(256, 512, StructureType.Compress, ActivationType.LeakyReLU),
				nn.Conv2d(512, 4000, 1)]
				#*get_basic_structure(512, 1024, StructureType.Compress, ActivationType.LeakyReLU),
				#nn.Conv2d(1024, 4000, 1)]

		else:
			self.base_modules_enc = [*get_basic_structure(3, 64, StructureType.Compress, ActivationType.LeakyReLU),
				*get_basic_structure(64, 64, StructureType.Compress, ActivationType.LeakyReLU),
				*get_basic_structure(64, 128, StructureType.Compress, ActivationType.LeakyReLU),
				*get_basic_structure(128, 256, StructureType.Compress, ActivationType.LeakyReLU),
				*get_basic_structure(256, 512, StructureType.Compress, ActivationType.LeakyReLU),
				nn.Conv2d(512, 4000, 1)]
				#*get_basic_structure(512, 1024, StructureType.Compress, ActivationType.LeakyReLU),
				#nn.Conv2d(1024, 4000, 1)]

		
		
		if self.output_full_image:
			# in this case we reconstruct full image size so architecture is fixed
			#self.base_modules_dec = [*get_basic_structure(4000, 1024, StructureType.Expand, ActivationType.ReLU),
			self.base_modules_dec = [*get_basic_structure(4000, 512, StructureType.Expand, ActivationType.ReLU),
			#*get_basic_structure(1024, 512, StructureType.Expand, ActivationType.ReLU),
			*get_basic_structure(512, 256, StructureType.Expand, ActivationType.ReLU),
			*get_basic_structure(256, 128, StructureType.Expand, ActivationType.ReLU),
			*get_basic_structure(128, 64, StructureType.Expand, ActivationType.ReLU),
			*get_basic_structure(64, 32, StructureType.Expand, ActivationType.ReLU), 
			nn.Conv2d(32, 3, 3, 1, 1)]
		else:
			# if we want to generate only a patch / resized image, we can optimize further.
			# we use the amount of layers needed to generate the wanted image size
			base_width = 512
			if cfg.TO_RESIZE:
				latent_size = 8 / (cfg.ORIG_IMAGE_SIZE / cfg.RESIZE_DIM)
			else:
				latent_size = 8
			# base_width = 1024
			# latent_size = 4
			self.base_modules_dec = [*get_basic_structure(4000, base_width, StructureType.Expand, ActivationType.ReLU)]
			for i in range(int(math.log2((output_size/2)/latent_size))):
				self.base_modules_dec.extend([*get_basic_structure(int(base_width/(2**i)), int(base_width/(2**(i+1))), StructureType.Expand, ActivationType.ReLU)])
			self.base_modules_dec.append(nn.Conv2d(int(base_width/(2**(i+1))), 3, 3, 1, 1))

		self.enc_model = nn.Sequential(*self.base_modules_enc)
		if cfg.TO_NORMALIZE:
			last_act = nn.Tanh()
		else:
			last_act = nn.Sigmoid()
		self.dec_model = nn.Sequential(*self.base_modules_dec, last_act) #nn.Sigmoid()) 

		# print(self.enc_model)
		# print(self.dec_model)

		#print(self.model)

	def forward(self, input):
		# latent = self.enc_model(input)
		# import pdb
		# pdb.set_trace()
		# return self.dec_model(latent)
		if self.extract_features:
			self.features = []
			#res = self.enc_model(input)
			res = input
			# for layer in self.enc_model:
			# 	res = layer(res)
			# 	if len(self.features) < 4:
			# 		#if isinstance(layer, nn.Conv2d):
			# 		if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
			# 			self.features.append(res)			
			#res = self.dec_model(res)
			res = self.enc_model(res)
			seen = 0
			for layer in self.dec_model:
				res = layer(res)
				#if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
				if isinstance(layer, nn.Tanh):
				#if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU) or isinstance(layer, nn.Tanh):
					# seen += 1
					# if seen > 1:
					self.features.append(res)
			return res
		else:
			return self.dec_model(self.enc_model(input))

	def get_encoder(self):
		return self.enc_model

	def get_decoder(self):
		return self.dec_model

	def load_pretrained_encoder(self, encoder_params_file_path):
		self.enc_model.load_state_dict(torch.load(encoder_params_file_path))

	def load_pretrained_decoder(self, decoder_params_file_path):
		self.dec_model.load_state_dict(torch.load(decoder_params_file_path))
		torch.nn.init.normal_(self.dec_model[-2].weight.data, mean=0.0, std=0.02)
		# self.dec_model[-4].weight.data.normal_(1.0, 0.02)
        # self.dec_model[-4].bias.data.fill_(0)
		# torch.nn.init.normal_(self.dec_model[-5].weight.data, mean=0.0, std=0.02)
		# import pdb
		# pdb.set_trace()

	def get_features(self):
		return self.features

	





# Discriminator network
class DiscriminatorNet(nn.Module):
	def __init__(self, input_full_image=False, input_size=128):
		super(DiscriminatorNet, self).__init__()

		self.input_full_image = input_full_image 
		self.input_size = input_size

		base_modules = [*get_basic_structure(3, 64, StructureType.Compress, ActivationType.LeakyReLU, normalize=False)] #norm=NormType.InstanceNorm)]
		# we generate same output grid size, so there's a different case for full image inserted and partial size
		if self.input_full_image:
			# full image - fixed arch
			base_modules.extend([*get_basic_structure(64, 128, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.BatchNorm), #InstanceNorm
			*get_basic_structure(128, 256, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.BatchNorm),
			*get_basic_structure(256, 512, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.BatchNorm),
			#*get_basic_structure(512, 1024, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.BatchNorm),
			#*get_basic_structure(1024, 2048, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.BatchNorm),
			#nn.Conv2d(2048, 1, 3, 1, 1)])
			nn.Conv2d(512, 1, 3, 1, 1)])
		else:
			# partial size - amount of layers changes according to input size
			for i in range(int(math.log2(input_size/8))):
				base_modules.extend([*get_basic_structure(64*(2**i), 128*(2**i), StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.BatchNorm)])	
			base_modules.append(nn.Conv2d(128*(2**i), 1, 3, 1, 1))

		self.model = nn.Sequential(*base_modules)

	def forward(self, input):
		return self.model(input)

	def load_model(self, model_params_file_path):
		self.model.load_state_dict(torch.load(model_params_file_path)) 

	# def get_model(self):
	# 	return self.model







class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

