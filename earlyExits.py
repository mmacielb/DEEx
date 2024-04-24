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

class EE_block(nn.Module):
	def __init__(self,input_shape,device):
		super(EE_block, self).__init__()

		self.input_shape = input_shape
		# self.total_neurons = 6*6*128
		self.total_neurons = 1152


		self.layers = nn.ModuleList()
		# self.classifier = nn.ModuleList()
		self.layers.append(nn.Conv2d(self.input_shape, 32, kernel_size=3, stride=1, padding=1).to(device))
		self.layers.append(nn.MaxPool2d(kernel_size=3).to(device))
		self.layers.append(nn.Dropout(p=0.5, inplace=False).to(device))
		self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6, 6)).to(device))
		self.classifier = nn.Sequential(nn.Linear(in_features=self.total_neurons, out_features=10, bias=True)).to(device)

		# self.conv = nn.Conv2d(self.input_shape, 32, kernel_size=3, stride=1, padding=1)
		# self.maxpool = nn.MaxPool2d(kernel_size=3)
		# self.dropout =  nn.Dropout(p=0.5, inplace=False).to(self.device)
		# self.adaptative = nn.AdaptiveAvgPool2d(output_size=(6, 6)).to(self.device)     ### Faz um pooling e coloca a saída no formato definido no outpusize
		# self.total_neurons = 6*6*128
		# self.linear = nn.Linear(in_features=self.total_neurons, out_features=10, bias=True).to(self.device)


	def forward(self, x):
		#branch = nn.ModuleList([self.conv,nn.ReLU(inplace=True),self.maxpool, self.dropout, self.adaptative, Flatten(), self.linear])
		for layer in self.layers:
			x = layer(x)
		x = x.view(x.size(0), -1)
		output = self.classifier(x)
		return output

class EarlyExitDNN(nn.Module):

	#def __init__(self, modelName, pretrained=True):
	def __init__(self, modelName, n_branches, position_list, n_classes, input_dim, device):

		super(EarlyExitDNN, self).__init__()

		self.modelName = modelName
		self.n_branches = n_branches
		self.position_list = position_list
		self.n_classes = n_classes
		self.input_dim = input_dim
		self.device = device

		self.stages = nn.ModuleList()
		self.exits = nn.ModuleList()
		self.layers = nn.ModuleList()
		self.classifier = nn.ModuleList()

		#print('device-',self.device)
		
		build_early_exit_dnn = self.dnn_architecture_model()
		build_early_exit_dnn()
		# print('BUILLLDDD \n',build_early_exit_dnn)
		# quit()


	def dnn_architecture_model(self):

		"""
		This method selects the backbone to insert the early exits.
		"""

		architecture_dnn_model_dict = {"alexnet": EarlyExitAlexnet(self.input_dim, self.device)} #,
									   # "mobilenet": self.early_exit_mobilenet,
									   # "efficientnet_b1": self.early_exit_efficientnet_b1}

		# self.pool_size = 7 if (self.model_name == "vgg16") else 1   				#### ver pq 7 ou 1
		return architecture_dnn_model_dict.get(self.modelName)#, self.invalid_model)



class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()
	
	def forward(self, x):
		x = x.view(x.size(0), -1)
		return x

class EarlyExitAlexnet(nn.Module):

	#def __init__(self):
	def __init__(self, input_dim, device):

		super(EarlyExitAlexnet, self).__init__()

		self.n_branches = 2
		self.position_list = [2,5]
		self.n_classes = 10
		self.input_dim = input_dim
		self.device = device

		# build_early_exit_dnn = self.dnn_architecture_model()
		# build_early_exit_dnn()

		self.stages = nn.ModuleList()
		self.exits = nn.ModuleList()
		self.layers = nn.ModuleList()
		# self.cost = []
		self.stage_id = 0

		# Loads the backbone model. In other words, Alexnet architecture provided by Pytorch.
		backbone_model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(self.device)
		# print('bb')
		# print(backbone_model)
		# print('--- --- ---')


		if self.n_branches > 3:
			print('the number of branches is greather then the layes in alexnet model')
			quit()

		conv_teste = nn.Conv2d(16, 33, 3, stride=2)

		for i, layer in enumerate(backbone_model.features):
			self.layers.append(layer)
			if type(layer) == type(conv_teste):
				self.input_shape = layer.out_channels
				# print('SHAPE',self.input_shape)
				# quit()
			if self.stage_id < len(self.position_list):
				# print('ENTREI 1',self.stage_id)
				if i == self.position_list[self.stage_id]:
					# print('ENTREI 2  --  ',type(layer))
					#self.exits.append(self.early_exit_block(self,n))
					self.stages.append(nn.Sequential(*self.layers))
					self.exits.append(EE_block(self.input_shape,self.device))
					self.layers = nn.ModuleList()
					self.stage_id += 1

		# print('AAAAAAAAAA --\n',self.stages)
		# print('AAAAAAAAAA --\n',*self.stages)
		# quit()
		

		self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6, 6)))				   ## coloca a saída no formato definido no outpusize. Esta fora do features e do classifier da backbone 
		self.stages.append(nn.Sequential(*self.layers))

		#Aqui a gente adiciona as camadas neurais que vão classificar, a partir dos atributos extraídos anteriormente.
		self.classifier = backbone_model.classifier
		self.classifier[1] = nn.Linear(9216, 4096)
		self.classifier[4] = nn.Linear(4096, 1024)
		self.classifier[6] = nn.Linear(1024, self.n_classes)		#Nº de camadas do dataset que se quer classificar.    
		self.softmax = nn.Softmax(dim=1)

		# print('STAGES \n',self.stages)
		# print("OOOOOOOOOO ",list(enumerate(self.exits)))
		# for i, stage in enumerate(self.exits):
		# 	print('\n----',i,stage)
		# 	print('++',self.stages[i])
		# quit()


	# def early_exit_block(self,n):

	# 	conv = nn.Conv2d(n, 32, kernel_size=3, stride=1, padding=1)

	# 	maxpool = nn.MaxPool2d(kernel_size=3)

	# 	dropout =  nn.Dropout(p=0.5, inplace=False).to(self.device)

	# 	adaptative = nn.AdaptiveAvgPool2d(output_size=(6, 6)).to(self.device)     ### Faz um pooling e coloca a saída no formato definido no outpusize

	# 	total_neurons = 6*6*128

	# 	linear = nn.Linear(in_features=total_neurons, out_features=10, bias=True).to(self.device)

	# 	#branch = nn.ModuleList([self.conv,nn.ReLU(inplace=True),self.maxpool, self.dropout, self.adaptative, Flatten(), self.linear])
	# 	branch = nn.ModuleList([conv,nn.ReLU(inplace=True),maxpool, dropout, adaptative, Flatten(), linear])

	# 	return branch



	def forward(self,x):
		output = [] #{i:[] for i in range(1,self.n_branches+2)}
		confidence = [] #{i:[] for i in range(1,self.n_branches+2)}
		infered_class = [] #{i:[] for i in range(1,self.n_branches+2)}

		for i, stage in enumerate(self.exits):
			# print('----',i,stage)
			# print(self.stages[i])
			# quit()
			x = self.stages[i](x)
			# res_branch = self.exits[i](res)
			res_branch = stage(x)
			res_branch = res_branch.to(self.device)
			confidence_branch, infered_class_branch = torch.max(self.softmax(res_branch), 1)
			output.append(res_branch)
			confidence.append(confidence_branch)
			infered_class.append(infered_class_branch)

			# output[i+1].append(res_branch)
			# confidence[i+1].append(confidence_branch)
			# infered_class[i+1].append(infered_class_branch)

		res = self.stages[-1](x)
		res = torch.flatten(res, 1)
		output_bb = self.classifier(res)
		output_bb = output_bb.to(self.device)

		confidence_bb, infered_class_bb = torch.max(self.softmax(output_bb), 1)
		#Confidence mede a confiança da predição e infered_calss aponta a classe inferida pela DNN
		output.append(output_bb)
		confidence.append(confidence_bb)
		infered_class.append(infered_class_bb)

		# output[self.n_branches+1].append(output_bb)
		# confidence[self.n_branches+1].append(confidence_bb)
		# infered_class[self.n_branches+1].append(infered_class_bb)
		return output, confidence, infered_class




# class EarlyExitMobilenet(nn.Module):



# class EarlyExitEfficientnet_b1(nn.Module):












# backbone_model = models.mobilenet_v2(pretrained = True)
# print(len(list(backbone_model.children())))
#print(backbone_model)
