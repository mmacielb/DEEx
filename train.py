import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import cv2
import os, sys, time, math, argparse
import functools
from PIL import Image
from itertools import product
import warnings

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

import tools
import earlyExits as ee
#from torchsummary import summary
from temperature_scaling_gpleiss import ModelWithTemperature


## chamar a rede para treinar
def trainModel(device,train_loader, valid_loader,model,criterion,optimize,epochs):

	train_time, train_loss_dict, train_acc_dict, valid_time, valid_loss_dict, valid_acc_dict = tools.initialize_train(model)

	for n_epochs in range(epochs):

		train_res = tools.run_epoch(device,train_loader,model,criterion,optimize,n_epochs,train=True)
		valid_res = tools.run_epoch(device,valid_loader,model,criterion,optimize,n_epochs,train=False)

		train_time.append(train_res[0].value())
		valid_time.append(valid_res[0].value())

		for i in range(1, (n_exits)+1):
			train_loss_dict[i].append(train_res[1].value())
			train_acc_dict[i].append(train_res[2].value())
			valid_loss_dict[i].append(valid_res[1].value())
			valid_acc_dict[i].append(valid_res[2].value())
	
	return train_time, train_loss_dict, train_acc_dict, valid_time, valid_loss_dict, valid_acc_dict



if __name__ == '__main__':

	#warnings.filterwarnings("ignore", category=UserWarning) 
	#path_data = '../data'
	sys.dont_write_bytecode = True

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	epochs = 1

	## Tx de apredizado
	lr = 0.001 ##ou 
	#lr=1.5e-4

	#Otimazador
	opt = 'SGD'
	# opt = 'adam'

	dataset = 'cifar10'

	bt_size = 128
	input_dim = 224
	n_classes = 10

	n_branchs = 2
	position_list = [2,5]

	modelName = "alexnet"

	classes_list, label_list,train_loader, valid_loader = tools.data_set(dataset,bt_size,input_dim,train=True)

	#model = ee.EarlyExitDNN(input_dim, device, pretrained=True)
	model = ee.EarlyExitDNN(modelName, n_branchs, position_list, n_classes, input_dim, device)
	model = model.to(device)
	# print('model!!!')
	# quit()

	n_exits = model.n_branchs + 1


	# Paremetros de configuracao da rede neural
	criterion, optimize = tools.parameter(model,lr,opt)
	criterion = criterion.to(device)
	softmax = nn.Softmax(dim=1)


	# Preparar o que eu quero de resutados

	train_time, train_loss_dict, train_acc_dict, valid_time, valid_loss_dict, valid_acc_dict = trainModel(device,train_loader, valid_loader,model,criterion,optimize,epochs)
















		#scaled_model = ModelWithTemperature(model)
		#scaled_model.set_temperature(valid_loader)


		## Treino para cada epoca
		#start_train_time = time.time()
