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
#from torchsummary import summary

# from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


def data_set(dataset,bt_size,train=True):

	mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	split_ratio = 0.1
	batch_size = batch_size


	input_dim = 224

	data_transform = transforms.Compose([
	    transforms.Resize(input_dim),
	    transforms.CenterCrop(input_dim),
	    transforms.ToTensor(),
	    transforms.Normalize(mean = mean, std = std),
	])


	if dataset == 'cifar10':
		if train:
			train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
			classes_list = train_set.classes
			label_list = list(train_set.class_to_idx.values())
		
			# This line defines the size of validation dataset.
			val_size = int(split_ratio*len(train_set))

			# This line defines the size of training dataset.
			train_size = int(len(train_set) - val_size)

			#This line splits the training dataset into train and validation, according split ratio provided as input.
			train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

			#This block creates data loaders for training, validation and test datasets.

			train_loader = DataLoader(train_dataset, batch_size = bt_size, shuffle=True, num_workers=2, pin_memory=True)
			val_loader = DataLoader(val_dataset, batch_size = bt_size_test, num_workers=2, pin_memory=True)


		else:	
			test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)
			classes_list = test_set.classes
			label_list = list(test_set.class_to_idx.values())

			#This block creates data loaders for training, validation and test datasets.
			test_loader = DataLoader(test_dataset, batch_size = bt_size_test, num_workers=2, pin_memory=True)



		# print(indices)
		# quit()




	if dataset == 'cifar100':
		train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=data_transform)
		test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transform)

		classes_list = train_set.classes

		label_list = list(train_set.class_to_idx.values())

		# This line defines the size of validation dataset.
		val_size = int(split_ratio*len(train_set))

		# This line defines the size of training dataset.
		train_size = int(len(train_set) - val_size)

		#This line splits the training dataset into train and validation, according split ratio provided as input.
		train_dataset, val_dataset = random_split(train_set, [train_size, val_size])


	if dataset == 'Caltech256':
		data_set = datasets.Caltech256(root='./data', download=True, transform=data_transform)

		# classes_list = data_set.classes
		# label_list = list(data_set.class_to_idx.values())

		indices = np.arange(len(train_set))

		np.random.shuffle(indices)


		teste = data_set[118]
		print(len(data_set))


	if train:
		return classes_list, label_list,train_loader, val_loader

	else:
		return classes_list, label_list, test_loader



def parameter(device,lr,opt):
	
	#lr=0.001 ##ou 
	#lr=1.5e-4


	#criterion = nn.NLLLoss()
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.to(device)

	#opt = 'SGD'
	# opt = 'adam'

	if opt == 'adam':
	  optimize_b1 = optim.Adam(branch1.parameters(),lr = lr)
	  optimize_b2 = optim.Adam(branch2.parameters(),lr = lr)
	  optimize_bb = optim.Adam(backbone.parameters(),lr = lr)

	else:
	  optimize_b1 = optim.SGD(branch1.parameters(),lr = 0.001,momentum=0.9)
	  optimize_b2 = optim.SGD(branch2.parameters(),lr = 0.001,momentum=0.9)
	  optimize_bb = optim.SGD(backbone.parameters(),lr = 0.001,momentum=0.9)

	