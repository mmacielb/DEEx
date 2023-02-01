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

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


def dataset(dataset):

	mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]


	### Pytorhch
	#mean=[0.485, 0.456, 0.40
	#std = [0.229, 0.224, 0.225]

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# device = torch.device('cpu')

	input_dim = 224

	data_transform = transforms.Compose([
	    transforms.Resize(input_dim),
	    transforms.CenterCrop(input_dim),
	    transforms.ToTensor(),
	    transforms.Normalize(mean = mean, std = std),
	])




	if dataset == 'cifar_10':
		train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
		test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)


	if dataset == 'cifar100':
		train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=data_transform)
		test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transform)

	if dataset == 'caltech_256':
		data_set = datasets.Caltech256(root='./data', download=True, transform=data_transform)

		classes_list = data_set.classes

		label_list = list(data_set.class_to_idx.values())

		return classes_list, label_list
		
a, b = datasets('cifar_10')

print(a,'\n',b)



def quadrado(numero):
	res = numero**2
	return res

teste = quadrado(4)
print(teste)


# def backbone_model(backbone_model,pretrained):


# 	backbone_model = models.backbone_model(pretrained = pretrained)

# 	print('ok!!')

# 	return backbone_model

# backbone_model = 'alexnet'
# pretrained = True

# backbone_model = backbone_model(backbone_model,pretrained)


	