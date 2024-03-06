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


def data_set(dataset,bt_size,input_dim,train):

	mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	split_ratio = 0.1
	batch_size = batch_size
	bt_size_test = batch_size


	input_dim = input_dim

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
			test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform)
			classes_list = test_set.classes
			label_list = list(test_set.class_to_idx.values())

			#This block creates data loaders for training, validation and test datasets.
			test_loader = DataLoader(test_dataset, batch_size = bt_size_test, num_workers=2, pin_memory=True)



		# print(indices)
		# quit()




	if dataset == 'cifar100':
		if train:
			train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=data_transform)

			classes_list = train_set.classes

			label_list = list(train_set.class_to_idx.values())

			# This line defines the size of validation dataset.
			val_size = int(split_ratio*len(train_set))

			# This line defines the size of training dataset.
			train_size = int(len(train_set) - val_size)

			#This line splits the training dataset into train and validation, according split ratio provided as input.
			train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

			train_loader = DataLoader(train_dataset, batch_size = bt_size, shuffle=True, num_workers=2, pin_memory=True)
			val_loader = DataLoader(val_dataset, batch_size = bt_size_test, num_workers=2, pin_memory=True)

		else:
			test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transform)


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



def parameter(model,lr,opt):
	
	#lr=0.001 ##ou 
	#lr=1.5e-4


	#criterion = nn.NLLLoss()
	criterion = nn.CrossEntropyLoss()

	#opt = 'SGD'
	# opt = 'adam'

	if opt == 'adam':
		optimize = optim.Adam(model.parameters(),lr = lr)

	else:
		optimize = optim.SGD(model.parameters(),lr = lr,momentum=0.9)

	return criterion, optimize

def initialize(classes_list,model):

	exits = model.n_branchs+1

	col = ['epoch']

	test_conf_matrix = np.zeros(len(classes_list)*len(classes_list)).reshape(len(classes_list),len(classes_list))

	all_conf_matrix = {}
	train_times = {}
	val_times = {}

	for i in range(1,exits+1):
		col.append('loss_'+str(i),'acc_'+str(i))
		all_conf_matrix[i] = test_conf_matrix
		train_times[i] = 0.0 
		val_times[i] = 0.0 


	train_res = {i: [] for i in col}
	val_res = {i: [] for i in col}
	#test_res = {i: [] for i in col}

	return


def initialize_epoch(classes_list,model,epochs):
	n_exits = model.n_branchs + 1




#	for epc in range(epochs):
		
	train_acc = {i: 0.0 for i in range(1, (n_exits)+1)}
	#train_acc_b2 = 0.0
	#train_acc_bb = 0.0
	val_acc = {i: 0.0 for i in range(1, (n_exits)+1)}
	#val_acc_b2 = 0.0
	#val_acc_b3 = 0.0
	
	running_loss_dict = {i: [] for i in range(1, (n_exits)+1)}
	#running_loss = []
	train_acc_dict = {i: [] for i in range(1, (n_exits)+1)}
	#train_acc_list = []
	val_loss_dict = {i: [] for i in range(1, (n_exits)+1)}
	#running_loss = []
	val_acc_dict = {i: [] for i in range(1, (n_exits)+1)}
	#train_acc_list = []

	return

class Meter():
    """
    A little helper class which keeps track of statistics during an epoch.
    """
    def __init__(self, name, cum=False):
        """
        name (str or iterable): name of values for the meter
            If an iterable of size n, updates require a n-Tensor
        cum (bool): is this meter for a cumulative value (e.g. time)
            or for an averaged value (e.g. loss)? - default False
        """
        self.cum = cum
        if type(name) == str:
            name = (name,)
        self.name = name

        self._total = torch.zeros(len(self.name))
        self._last_value = torch.zeros(len(self.name))
        self._count = 0.0

    def update(self, data, n=1):
        """
        Update the meter
        data (Tensor, or float): update value for the meter
            Size of data should match size of ``name'' in the initialized args
        """
        self._count = self._count + n
        if torch.is_tensor(data):
            self._last_value.copy_(data)
        else:
            self._last_value.fill_(data)
        self._total.add_(self._last_value)

    def value(self):
        """
        Returns the value of the meter
        """
        if self.cum:
            return self._total
        else:
            return self._total / self._count

    def __repr__(self):
        return '\t'.join(['%s: %.5f (%.3f)' % (n, lv, v)
            for n, lv, v in zip(self.name, self._last_value, self.value())])