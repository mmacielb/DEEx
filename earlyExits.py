import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import os, cv2, sys, time, math
import os, sys, time, math, argparse
import functools
from PIL import Image
from itertools import product

import torch, random
import torchvision

from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from tqdm.notebook import tqdm, trange

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from torchvision import transforms, utils, datasets

#import tools
#from torchsummary import summary


class EarlyExitDNN(nn.Module):

	def __init__(self, modelName, pretrained=True):
	#def __init__(self, modelName, n_branchs, position_list, n_classes, input_dim, device, pretrained=True):

		super(EarlyExitDNN, self).__init__()

		self.modelName = modelName
		self.n_branchs = n_branchs
		self.position_list = position_list
		self.n_classes = n_classes
		self.input_dim = input_dim
		self.device = device
		self.pretrained = pretrained

		# build_early_exit_dnn = self.dnn_architecture_model()
		# build_early_exit_dnn()


	def dnn_architecture_model(self):

		"""
		This method selects the backbone to insert the early exits.
		"""

		architecture_dnn_model_dict = {"alexnet": self.early_exit_alexnet} #,
									   # "mobilenet": self.early_exit_mobilenet,
									   # "efficientnet_b1": self.early_exit_efficientnet_b1}

		# self.pool_size = 7 if (self.model_name == "vgg16") else 1   				#### ver pq 7 ou 1
		# return architecture_dnn_model_dict.get(self.model_name, self.invalid_model)


	def flatten(self,input):
		return input.view(input.shape[0],-1)


	def early_exit_block_alexnet(self,n):

		conv = lambda n: [nn.Conv2d(n, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)]

		maxpool = nn.MaxPool2d(kernel_size=3)

		dropout =  nn.Dropout(p=0.5, inplace=False).to(device)

		adaptative = nn.AdaptiveAvgPool2d(output_size=(6, 6)).to(device)     ### Faz um pooling e coloca a saída no formato definido no outpusize

    total_neurons = 6*6*128

    linear = nn.Linear(in_features=total_neurons, out_features=10, bias=True).to(device)

    branch = conv(n)+maxpool+dropout+adaptative+flatten()+linear

    return branch


		
	def early_exit_alexnet(self):
		"""
		This method inserts early exits into a Alexnet model
		"""

		self.stages = nn.ModuleList()
		self.exits = nn.ModuleList()
		self.layers = nn.ModuleList()
		self.cost = []
		self.stage_id = 0

		# Loads the backbone model. In other words, Alexnet architecture provided by Pytorch.
		backbone_model = models.alexnet(self.pretrained)

		# print(backbone_model)

		if self.n_branchs > 3:
			print('the number of branchs is greather then the layes in alexnet model')
			quit()

		conv_teste = nn.Conv2d(16, 33, 3, stride=2)

		for i, layer in enumarate(backbone_model.features):
			if type(layer) == type(conv_teste):
				n = layer.output_channels
			if i == position_list[self.stage_id]:
				sel.exits.append(early_exit_block_alexnet(self,n))
				self.stage_id += 1

		self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6, 6)))				### coloca a saída no formato definido no outpusize. Esta fora do features e do classifier da backbone 
		self.stages.append(nn.Sequential(*self.layers))

		#Aqui a gente adiciona as camadas neurais que vão classificar, a partir dos atributos extraídos anteriormente.
		self.classifier = backbone_model.classifier
		self.classifier[1] = nn.Linear(9216, 4096)
		self.classifier[4] = nn.Linear(4096, 1024)
		self.classifier[6] = nn.Linear(1024, n_classes) #Nº de damdas do dataset que se quer classificar.    
		self.softmax = nn.Softmax(dim=1)

		# # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
		# self.threshold_flop_list = self.where_insert_early_exits()

		# for layer in backbone_model.features:
		#   self.layers.append(layer)
		#   if (isinstance(layer, nn.ReLU)) and (self.is_suitable_for_exit()):
		#     self.add_exit_block()



		# self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6, 6)))
		# self.stages.append(nn.Sequential(*self.layers))


		# self.classifier = backbone_model.classifier
		# self.classifier[6] = nn.Linear(in_features=4096, out_features=self.n_classes, bias=True)
		# self.softmax = nn.Softmax(dim=1)
		# self.set_device()

# backbone_model = models.mobilenet_v2(pretrained = True)

# print(len(list(backbone_model.features.children())))
# print(backbone_model)

backbone_model = models.alexnet(pretrained = True)

# print(len(list(backbone_model.children())))
print(backbone_model)




#  def add_exit_block(self):
#     """
#     This method adds an early exit in the suitable position.
#     """
#     input_tensor = torch.rand(1, self.channel, self.width, self.height)

#     self.stages.append(nn.Sequential(*self.layers))
#     x = torch.rand(1, 3, 224, 224)#.to(self.device)
#     feature_shape = nn.Sequential(*self.stages)(x).shape
#     self.exits.append(EarlyExitBlock(feature_shape, self.pool_size, self.n_classes, self.exit_type, self.device))#.to(self.device))
#     self.layers = nn.ModuleList()
#     self.stage_id += 1    


# class EarlyExitBlock(nn.Module):
#   """
#   This EarlyExitBlock allows the model to terminate early when it is confident for classification.
#   """
#   def __init__(self, input_shape, pool_size, n_classes, exit_type, device):
#     super(EarlyExitBlock, self).__init__()
#     self.input_shape = input_shape

#     _, channel, width, height = input_shape
#     self.expansion = width * height if exit_type == 'plain' else 1

#     self.layers = nn.ModuleList()

#     if (exit_type == 'bnpool'):
#       self.layers.append(nn.BatchNorm2d(channel))

#     if (exit_type != 'plain'):
#       self.layers.append(nn.AdaptiveAvgPool2d(pool_size))
    
#     #This line defines the data shape that fully-connected layer receives.
#     current_channel, current_width, current_height = self.get_current_data_shape()

#     self.layers = self.layers#.to(device)

#     #This line builds the fully-connected layer
#     self.classifier = nn.Sequential(nn.Linear(current_channel*current_width*current_height, n_classes))#.to(device)

#     self.softmax_layer = nn.Softmax(dim=1)



