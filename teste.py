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

# def testModel(device,test_loader,model,criterion,optimize,weight,p_min_list,scaler):

# 	test_time, test_loss_dict, test_acc_dict,test_conf_dict,all_conf_matrix = tools.initialize_test(model,classes_list) #tools

# 	for p_min in p_min_list:
# 		print(p_min)
# 		samples_list = [0,0,0]

# 		test_time_meter, test_loss_epoch, test_acc_epoch, test_conf_epoch = tools.run_epoch(device,test_loader,model,criterion,optimize,weight,p_min,scaler,train=False)

# 		test_time.append(test_time_meter)
# 		for k,v in test_loss_dict.items():
# 			test_loss_dict[k].append(test_loss_epoch[k])
# 			test_acc_dict[k].append(test_acc_epoch[k])
# 			if k != 'model':
# 				test_conf_dict[k].append(test_conf_epoch[k])
# 				samples_list[k-1].append(len([x for x in test_conf_epoch[k] if not math.isnan(x)]))

# 		return test_time_meter, test_loss_epoch, test_acc_epoch, test_conf_epoch

def testModel(device,test_loader,model,criterion,optimize,weight,epochs,scaler):

	test_time, test_loss_dict, test_acc_dict,test_conf_dict, _, _, _, _ = tools.initialize_train(model) #tools

	for n_epochs in range(epochs):
		print(n_epochs)

		test_time_meter, test_loss_epoch, test_acc_epoch, test_conf_epoch = tools.run_epoch(device,test_loader,model,criterion,optimize,weight,n_epochs,scaler,train=False) #tools
		# lr_scheduler.step()

		test_time.append(test_time_meter)

		for k,v in test_loss_dict.items():
			test_loss_dict[k].append(test_loss_epoch[k])
			test_acc_dict[k].append(test_acc_epoch[k])
			if k != 'model':
				test_conf_dict[k].append(test_conf_epoch[k])

	return test_time, test_loss_dict, test_acc_dict,test_conf_dict


if __name__ == '__main__':

	torch.cuda.empty_cache()
	sys.dont_write_bytecode = True

	path_model = '../trainedModels/'
	path_result =  '../results/'

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('device: ',device,'\n')

	p_min_list = [i/100 for i in range(0,105,5)]

	epochs = 21

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

	rod = 2

	test_time, test_loss_dict, test_acc_dict,test_conf_dict = testModel(device,test_loader, model,criterion,optimizer,weight,rod,scaler)

	print('Fim do Teste')

	torch.cuda.empty_cache()
	
	##### Editando os Resultados
	rod_list = [ i for i in range(1,rod+1)]
	rod_dict = {'epoch':rod_list}
	time_result = {'epoch':rod_list,'test':test_time}
	print(time_result)
	# quit()

	test_loss_dict = rod_dict | test_loss_dict
	test_acc_dict = rod_dict | test_acc_dict
	test_conf_dict = rod_dict | test_conf_dict

	#### Salvando os Resultados
	time_pd = pd.DataFrame.from_dict(time_result)
	time_pd.to_csv(path_or_buf = path_result+'time-test-res-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)

	test_loss_pd = pd.DataFrame.from_dict(test_loss_dict)
	test_loss_pd.to_csv(path_or_buf = path_result+'test-res-loss-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)

	test_acc_pd = pd.DataFrame.from_dict(test_acc_dict)
	test_acc_pd.to_csv(path_or_buf = path_result+'test-res-acc-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)

	# print(test_conf_dict)
	test_conf_pd = pd.DataFrame.from_dict(test_conf_dict)
	test_conf_pd.to_csv(path_or_buf = path_result+'test-res-conf-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)


	# print(test_loss_dict)
	print('Fim!')