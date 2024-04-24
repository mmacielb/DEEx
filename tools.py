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


def data_set(dataset,bt_size,input_dim,train):

	mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	split_ratio = 0.1
	batch_size = bt_size
	bt_size_test = bt_size


	input_dim = input_dim

	data_transform = transforms.Compose([
	    transforms.Resize(input_dim),
	    transforms.CenterCrop(input_dim),
	    transforms.ToTensor(),
	    transforms.Normalize(mean = mean, std = std),
	])


	if dataset == 'cifar10':
		if train:
			train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=data_transform)
			classes_list = train_set.classes
			label_list = list(train_set.class_to_idx.values())
		
			# This line defines the size of validation dataset.
			val_size = int(split_ratio*len(train_set))

			# This line defines the size of training dataset.
			train_size = int(len(train_set) - val_size)

			#This line splits the training dataset into train and validation, according split ratio provided as input.
			train_dataset, val_dataset = random_split(train_set, [train_size, val_size])
			# print(val_dataset[0])
			# quit()
			torch.save(val_dataset, "../data/valid_set.pt")

			#This block creates data loaders for training, validation and test datasets.

			train_loader = DataLoader(train_dataset, batch_size = bt_size, shuffle=True, num_workers=2, pin_memory=True)
			val_loader = DataLoader(val_dataset, batch_size = bt_size_test, num_workers=2, pin_memory=True)


		else:	
			test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=data_transform)
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



def parameter(model,lr,opt,n_branches):
	
	#lr=0.001 ##ou 
	#lr=1.5e-4
	#criterion = nn.NLLLoss()
	#opt = 'SGD'
	# opt = 'adam'

	criterion = nn.CrossEntropyLoss()
	lr = lr
	opt = opt


	if opt == 'adam':
		optimizer = optim.Adam(model.parameters(),lr = lr)

	else:
		#print(list(model.parameters()))
		#quit()
		#optimizer = optim.SGD(model.parameters(),lr = lr,momentum=0.9)

		optimizer = optim.SGD([{'params': model.stages.parameters(), 'lr': lr}, 
		{'params': model.exits.parameters(), 'lr': lr},
		{'params': model.classifier.parameters(), 'lr': lr}], momentum=0.9) #weight_decay=args.weight_decay)

		# weight = np.linspace(0.3, 1, n_branches+1)
		# weight = np.linspace(1, 0.3, n_branches+1)
		weight = np.ones(n_branches+1)


	return criterion, optimizer, weight

def initialize(classes_list,model):

	exits = model.n_branches+1

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


def initialize_train(model):
	n_exits = model.n_branches + 1
	
	train_time = []
	valid_time = []

	train_loss_dict = {i: [] for i in range(1, (n_exits)+1)}
	train_loss_dict['model']=[]
	train_acc_dict = {i: [] for i in range(1, (n_exits)+1)}
	train_acc_dict['model']=[]
	train_conf_dict = {i: [] for i in range(1, (n_exits)+1)}
	# train_conf_dict['model']=[]

	valid_loss_dict = {i: [] for i in range(1, (n_exits)+1)}
	valid_loss_dict['model']=[]
	valid_acc_dict = {i: [] for i in range(1, (n_exits)+1)}
	valid_acc_dict['model']=[]
	valid_conf_dict = {i: [] for i in range(1, (n_exits)+1)}
	# valid_conf_dict['model']=[]

	return train_time, train_loss_dict, train_acc_dict,train_conf_dict, valid_time, valid_loss_dict, valid_acc_dict,valid_conf_dict

def compute_metrics(criterion,weight_list,output_list,confidence_list,class_list, target):
	model_loss = 0.0
	ee_loss = [] #{i:[] for i in range(1, (n_exits)+1)}
	ee_acc = [] #{i:[] for i in range(1, (n_exits)+1)}
	ee_conf = []

	for i, (output,inf_class,weight,confidence) in enumerate(zip(output_list, class_list,weight_list,confidence_list), 1):
		loss_branch = criterion(output, target)
		model_loss += weight*loss_branch

		acc_branch = accuracy_score(inf_class.cpu(),target.cpu())
		ee_acc.append(acc_branch)
		ee_loss.append(loss_branch.item())
		ee_conf.append(torch.mean(confidence).item())
		# print('AAAAAA\n',ee_conf)

		#loss_branch.backward()

	model_acc = np.mean(np.array(ee_acc))
	return model_loss,model_acc,ee_loss,ee_acc,ee_conf

def run_epoch(device,loader,model,criterion,optimizer,weight,n_epochs,scaler,train=True):
	'''
	Inicializa as variaveis locais
	realiza a rodada de uma época
	retorna o tempo (time.value()) e os valores de loss e acuracia (met.[i].value()) da rodada
	'''
	n_exits = model.n_branches + 1

	time_list_epoch = []
	model_loss_list, model_acc_list = [], []
	ee_loss_list, ee_acc_list = [], []
	ee_conf_list = []
	loss_epoch = {}
	acc_epoch = {}
	confidence_epoch = {}

	if train:
		model.train()
		print('Training')
	else:
		model.eval()
		print('Evaluating')

	end = time.time()
	## comeca a rodada
	for i, (input, target) in enumerate(loader):
		# print(i," - finalmente!!!")
		if train:
			model.zero_grad()
			optimizer.zero_grad()

			# Forward pass
			input, target = input.to(device), target.to(device)
			with torch.cuda.amp.autocast(enabled=True):
				output_list,confidence_list,infered_class = model(input)	#Recebe o resultado da saída da rede em treinamento (3 listas)
				model_loss,model_acc,ee_loss,ee_acc,ee_conf = compute_metrics(criterion, weight, output_list,confidence_list,infered_class,target)	#Calcula a loss e acc dos resultados obtidos
			# print(output[1][0])
			# print('-----')
			# print(confidence)

			# Backward pass
			scaler.scale(model_loss).backward()

			scaler.step(optimizer)
			scaler.update()

			# loss.backward()
			# optimizer.step()
			# optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1

		else:
			with torch.no_grad():
				# Forward pass
				input, target = input.to(device), target.to(device)
				output_list,confidence_list, infered_class = model(input)	#Recebe o resultado da saída da rede em treinamento (3 listas)
				model_loss,model_acc,ee_loss,ee_acc,ee_conf = compute_metrics(criterion, weight, output_list,confidence_list,infered_class,target)	#Calcula a loss e acc dos resultados obtidos


		# #_, predictions = torch.topk(output, 1)
		# for i in range(1, (n_exits)+1):
		# 	acc.append(accuracy_score(infered_class[i].cpu(),target.cpu()))
		batch_time = time.time() - end
		time_list_epoch.append(batch_time)

		# print(ee_acc)
		# print('2-----2')
		# print(confidence)

		model_loss_list.append(model_loss.item())
		model_acc_list.append(model_acc)
		ee_loss_list.append(ee_loss)
		ee_acc_list.append(ee_acc)
		ee_conf_list.append(ee_conf)


	time_meter = round(np.mean(time_list_epoch), 4)
	loss_epoch['model'] = round(np.mean(model_loss_list), 4)
	acc_epoch['model'] =  round(np.mean(model_acc_list), 4)
	list_loss = np.mean(ee_loss_list, axis=0)
	list_acc = np.mean(ee_acc_list, axis=0)
	list_conf = np.mean(ee_conf_list, axis=0)
	for i in range(1, (n_exits)+1):
		loss_epoch[i] = list_loss[i-1]
		acc_epoch[i] = list_acc[i-1]
		confidence_epoch[i] = round(list_conf[i-1],4)

	### Colocar um print com valores da rodada
	print('epoch: ',n_epochs,'Model Loss = ',loss_epoch['model'],' Model Accuracy = ',acc_epoch['model'],'\n')

	return time_meter, loss_epoch, acc_epoch, confidence_epoch