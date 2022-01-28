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
	IntanceNorm = 1



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



class GeneratorNet(nn.Module):
	def __init__(self):
		super(GeneratorNet, self).__init__()
		self.model = [ #nn.Sequential(
            get_basic_structure(3, 64, StructureType.Compress, ActivationType.LeakyReLU),
            get_basic_structure(64, 64, StructureType.Compress, ActivationType.LeakyReLU),
			get_basic_structure(64, 128, StructureType.Compress, ActivationType.LeakyReLU),
			get_basic_structure(128, 256, StructureType.Compress, ActivationType.LeakyReLU),
			get_basic_structure(256, 512, StructureType.Compress, ActivationType.LeakyReLU),
            nn.Conv2d(512, 4000, 1),
			get_basic_structure(4000, 512, StructureType.Expand, ActivationType.ReLU),
            get_basic_structure(512, 256, StructureType.Expand, ActivationType.ReLU),
			get_basic_structure(256, 128, StructureType.Expand, ActivationType.ReLU),
			get_basic_structure(128, 64, StructureType.Expand, ActivationType.ReLU),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        ]#)

	def forward(self, input):
		#return self.model(x)
		x = input
		for stage in self.model:
			x = self.model(x)





class DiscriminatorNet(nn.Module):
	def __init__(self):
		super(DiscriminatorNet, self).__init__()
		self.model = [ #nn.Sequential(

		]

	def forward(self, input):
		#return self.model(x)
		x = input
		for stage in self.model:
			x = self.model(x)