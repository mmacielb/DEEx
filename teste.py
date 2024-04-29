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
# from temperature_scaling_gpleiss import ModelWithTemperature

def testModel(device,test_loader,model,criterion,optimize,weight,epochs,scaler):

	# train_time, train_loss_dict, train_acc_dict,train_conf_dict, valid_time, valid_loss_dict, valid_acc_dict,valid_conf_dict = tools.initialize_train(model) #tools

	for n_epochs in range(epochs):
		print(n_epochs)

		test_time_meter, test_loss_epoch, test_acc_epoch, test_conf_epoch = tools.run_epoch(device,test_loader,model,criterion,optimize,weight,n_epochs,scaler,train=False)





if __name__ == '__main__':

	torch.cuda.empty_cache()
	sys.dont_write_bytecode = True

	path_model = '../trainedModels/'
	path_result =  '../results/'

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('device: ',device,'\n')

	epochs = 200

	## Tx de apredizado
	lr = 0.001 ##ou 
	# lr=1.5e-4
	# lr = 0.01
	lr_warmup_epochs = 5


	#Otimazador
	opt = 'SGD'
	# opt = 'adam'
	scaler = torch.cuda.amp.GradScaler()

	dataset = 'cifar10'

	# bt_size = 128
	bt_size = 1
	input_dim = 224
	n_classes = 10

	n_branches = 2
	position_list = [2,5]


	modelName = "alexnet"

	classes_list, label_list, test_loader = tools.data_set(dataset,bt_size,input_dim,train=False) #tools

	model = ee.EarlyExitAlexnet(input_dim, device)
	model = model.to(device)
	# print('model!!!')
	# quit()

	model.load_state_dict(torch.load(path_model+modelName+"-"+opt+"-"+str(epochs)+"-epc.pt"))
	
	n_exits = model.n_branches + 1
	# Paremetros de configuracao da rede neural
	criterion, optimizer, weight = tools.parameter(model,lr,opt,n_branches) #tools
	criterion = criterion.to(device)
	softmax = nn.Softmax(dim=1)


	main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - lr_warmup_epochs, eta_min=0)


	warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=0.01, total_iters=lr_warmup_epochs)

	lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[lr_warmup_epochs])

	test_time, test_loss_dict, test_loss_dict,test_conf_dict = testModel(device,test_loader, model,criterion,optimizer,weight,epochs,scaler,lr_scheduler)

	torch.cuda.empty_cache()
	
	##### Editando os Resultados
	epochs_list = [ i for i in range(1,epochs+1)]
	epochs_dict = {'epoch':epochs_list}
	time_result = {'epoch':epochs_list,'train':test_time}

	test_loss_dict = epochs_dict | test_loss_dict
	test_loss_dict = epochs_dict | test_loss_dict
	test_conf_dict = epochs_dict | test_conf_dict

	# print(test_loss_dict)
	# print('____________')