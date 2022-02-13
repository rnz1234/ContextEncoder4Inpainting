import torch.nn as nn
import torch.nn.functional as F
import torch

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
	def __init__(self, output_full_image=False):
		super(GeneratorNet, self).__init__()

		self.output_full_image = output_full_image

		self.base_modules_enc = [*get_basic_structure(3, 64, StructureType.Compress, ActivationType.LeakyReLU),
            *get_basic_structure(64, 64, StructureType.Compress, ActivationType.LeakyReLU),
			*get_basic_structure(64, 128, StructureType.Compress, ActivationType.LeakyReLU),
			*get_basic_structure(128, 256, StructureType.Compress, ActivationType.LeakyReLU),
			*get_basic_structure(256, 512, StructureType.Compress, ActivationType.LeakyReLU),
            nn.Conv2d(512, 4000, 1)]

		self.base_modules_dec = [*get_basic_structure(4000, 512, StructureType.Expand, ActivationType.ReLU),
            *get_basic_structure(512, 256, StructureType.Expand, ActivationType.ReLU),
			*get_basic_structure(256, 128, StructureType.Expand, ActivationType.ReLU),
			*get_basic_structure(128, 64, StructureType.Expand, ActivationType.ReLU)]

		if self.output_full_image:
			self.base_modules_dec.extend([*get_basic_structure(64, 32, StructureType.Expand, ActivationType.ReLU), 
									nn.Conv2d(32, 3, 3, 1, 1)])
		else:
			self.base_modules_dec.append(nn.Conv2d(64, 3, 3, 1, 1))

		self.enc_model = nn.Sequential(*self.base_modules_enc)
		self.dec_model = nn.Sequential(*self.base_modules_dec, nn.Sigmoid())

		#print(self.model)

	def forward(self, input):
		return self.dec_model(self.enc_model(input))

	def get_encoder(self, input):
		return self.enc_model

	def get_decoder(self, input):
		return self.dec_model

	def load_pretrained_encoder(self, encoder_params_file_path):
		self.enc_model.load_state_dict(torch.load(encoder_params_file_path))

	def load_pretrained_decoder(self, decoder_params_file_path):
		self.dec_model.load_state_dict(torch.load(decoder_params_file_path))
	





# Discriminator network
class DiscriminatorNet(nn.Module):
	def __init__(self, output_full_image=False):
		super(DiscriminatorNet, self).__init__()

		self.output_full_image = output_full_image 

		base_modules = [*get_basic_structure(3, 64, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.InstanceNorm),
			*get_basic_structure(64, 128, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.InstanceNorm),
			*get_basic_structure(128, 256, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.InstanceNorm),
			*get_basic_structure(256, 512, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.InstanceNorm),]
		
		if self.output_full_image:
			base_modules.extend([*get_basic_structure(512, 1024, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.InstanceNorm),
			*get_basic_structure(1024, 2048, StructureType.Compress, ActivationType.LeakyReLU, norm=NormType.InstanceNorm),
			nn.Conv2d(2048, 1, 3, 1, 1)])
		else:
			base_modules.append(nn.Conv2d(512, 1, 3, 1, 1))

		self.model = nn.Sequential(*base_modules)

	def forward(self, input):
		return self.model(input)

	# def load_model(self, model_params_file_path):
	# 	self.model.load_state_dict(torch.load(model_params_file_path)) 

	# def get_model(self):
	# 	return self.model