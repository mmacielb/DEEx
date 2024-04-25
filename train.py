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


## chamar a rede para treinar
def trainModel(device,train_loader, valid_loader,model,criterion,optimize,weight,epochs,scaler):

	train_time, train_loss_dict, train_acc_dict,train_conf_dict, valid_time, valid_loss_dict, valid_acc_dict,valid_conf_dict = tools.initialize_train(model) #tools

	for n_epochs in range(epochs):
		print(n_epochs)

		train_time_meter, train_loss_epoch, train_acc_epoch, train_conf_epoch = tools.run_epoch(device,train_loader,model,criterion,optimize,weight,n_epochs,scaler,train=True) #tools
		valid_time_meter, valid_loss_epoch, valid_acc_epoch, valid_conf_epoch = tools.run_epoch(device,valid_loader,model,criterion,optimize,weight,n_epochs,scaler,train=False) #tools
		train_time.append(train_time_meter)
		valid_time.append(valid_time_meter)

		for k,v in train_loss_dict.items():
			train_loss_dict[k].append(train_loss_epoch[k])
			train_acc_dict[k].append(train_acc_epoch[k])
			valid_loss_dict[k].append(valid_loss_epoch[k])
			valid_acc_dict[k].append(valid_acc_epoch[k])
			if k != 'model':
				train_conf_dict[k].append(train_conf_epoch[k])
				valid_conf_dict[k].append(valid_conf_epoch[k])

	return train_time, train_loss_dict, train_acc_dict,train_conf_dict, valid_time, valid_loss_dict, valid_acc_dict,valid_conf_dict



if __name__ == '__main__':

	torch.cuda.empty_cache()
	#warnings.filterwarnings("ignore", category=UserWarning) 
	#path_data = '../data'
	sys.dont_write_bytecode = True

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('device: ',device,'\n')

	epochs = 90

	## Tx de apredizado
	# lr = 0.001 ##ou 
	# lr=1.5e-4
	lr = 0.1


	#Otimazador
	opt = 'SGD'
	# opt = 'adam'
	scaler = torch.cuda.amp.GradScaler()

	dataset = 'cifar10'

	# bt_size = 128
	bt_size = 32
	input_dim = 224
	n_classes = 10

	n_branches = 2
	position_list = [2,5]

	modelName = "alexnet"

	classes_list,label_list,train_loader,valid_loader = tools.data_set(dataset,bt_size,input_dim,train=True) #tools

	#model = ee.EarlyExitDNN(input_dim, device, pretrained=True)
	# model = EarlyExitDNN(modelName, n_branches, position_list, n_classes, input_dim, device) #ee
	model = ee.EarlyExitAlexnet(input_dim, device)
	model = model.to(device)
	# print('model!!!')
	# quit()

	n_exits = model.n_branches + 1


	# Paremetros de configuracao da rede neural
	criterion, optimize, weight = tools.parameter(model,lr,opt,n_branches) #tools
	criterion = criterion.to(device)
	softmax = nn.Softmax(dim=1)


	# Preparar o que eu quero de resutados

	train_time, train_loss_dict, train_acc_dict,train_conf_dict, valid_time, valid_loss_dict, valid_acc_dict,valid_conf_dict = trainModel(device,train_loader, valid_loader,model,criterion,optimize,weight,epochs,scaler)

	torch.cuda.empty_cache()
	
	##### Editando os Resultados
	epochs_list = [ i for i in range(1,epochs+1)]
	epochs_dict = {'epoch':epochs_list}
	time_result = {'epoch':epochs_list,'train':train_time,'valid':valid_time}

	train_loss_dict = epochs_dict | train_loss_dict
	train_acc_dict = epochs_dict | train_acc_dict
	train_conf_dict = epochs_dict | train_conf_dict
	valid_loss_dict = epochs_dict | valid_loss_dict
	valid_acc_dict = epochs_dict | valid_acc_dict
	valid_conf_dict = epochs_dict | valid_conf_dict

	# print(train_loss_dict)
	# print('____________')

	# path_model = '/content/drive/MyDrive/DEEx-Mariana/trainedModels/'
	# path_result =  '/content/drive/MyDrive/DEEx-Mariana/results/'
	path_model = '../trainedModels/'
	path_result =  '../results/'

	### Salvando a rede Treinada para o teste
	torch.save(model.state_dict(), path_model+modelName+"-"+opt+"-"+str(epochs)+"-epc.pt")

	#### Salvando os Resultados
	train_loss_pd = pd.DataFrame.from_dict(train_loss_dict)
	train_loss_pd.to_csv(path_or_buf = path_result+'train-res-loss-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)

	train_acc_pd = pd.DataFrame.from_dict(train_acc_dict)
	train_acc_pd.to_csv(path_or_buf = path_result+'train-res-acc-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)

	# print(train_conf_dict)
	train_conf_pd = pd.DataFrame.from_dict(train_conf_dict)
	train_conf_pd.to_csv(path_or_buf = path_result+'train-res-conf-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)

	valid_loss_pd = pd.DataFrame.from_dict(valid_loss_dict)
	valid_loss_pd.to_csv(path_or_buf = path_result+'valid-res-loss-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)

	valid_acc_pd = pd.DataFrame.from_dict(valid_acc_dict)
	valid_acc_pd.to_csv(path_or_buf = path_result+'valid-res-acc-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)

	valid_conf_pd = pd.DataFrame.from_dict(valid_conf_dict)
	valid_conf_pd.to_csv(path_or_buf = path_result+'valid-res-conf-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)

	time_pd = pd.DataFrame.from_dict(time_result)
	time_pd.to_csv(path_or_buf = path_result+'time-res-bAlexnet-02-'+opt+'-lr'+str(lr)+"-"+str(epochs)+'-epc.csv',sep="\t", index=False)

	torch.cuda.empty_cache()
	print('Fim do Treino!!!')















		#scaled_model = ModelWithTemperature(model)
		#scaled_model.set_temperature(valid_loader)


		## Treino para cada epoca
		#start_train_time = time.time()
