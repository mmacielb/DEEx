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

import tools
import earlyExits as ee
#from torchsummary import summary
from temperature_scaling_gpleiss import ModelWithTemperature



if __name__ == '__main__':

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	epochs = 100

	## Tx de apredizado
	lr = 0.001 ##ou 
	#lr=1.5e-4

	#Otimazador
	opt = 'SGD'
	# opt = 'adam'

	dataset = 'cifar10'

	bt_size = 128
	input_dim = 224


	classes_list, label_list,train_loader, valid_loader = tools.data_set(dataset,bt_size,input_dim,train=True)

	model = ee.EarlyExitDNN(input_dim, device, pretrained=True)
	model = model.to(device)


	# Paremetros de configuracao da rede neural
	criterion, optimize = tools.parameter(model,lr,opt)
	criterion = criterion.to(device)
	softmax = nn.Softmax(dim=1)


	# Preparar o que eu quero de resutados






	## chamar a rede para treinar
	def run_epoch(loader, model, criterion, optimizer, epoch=0, n_epochs=0, train=True):
		time_meter = Meter(name='Time', cum=True)
		loss_meter = Meter(name='Loss', cum=False)
		error_meter = Meter(name='Error', cum=False)

		if train:
			model.train()
			print('Training')
		else:
			model.eval()
			print('Evaluating')

		end = time.time()
		for i, (input, target) in enumerate(loader):
			if train:
				model.zero_grad()
				optimizer.zero_grad()

				# Forward pass
				input, target = input.to(device), target.to(device)
				output = model(input)
				loss = criterion(output, target)

				# Backward pass
				loss.backward()
				optimizer.step()
				optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1

			else:
				with torch.no_grad():
					# Forward pass
					input, target = input.to(device), target.to(device)
					output,confidence, infered_class = model(input)
					loss = criterion(output, target)

			# Accounting
			#_, predictions = torch.topk(output, 1)
			error = 1 - torch.eq(infered_class, target).float().mean()
			batch_time = time.time() - end
			end = time.time()

			# Log errors
			time_meter.update(batch_time)
			loss_meter.update(loss)
			error_meter.update(error)
			print('  '.join([
				'%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
					epoch, n_epochs, i + 1, len(loader)),
				str(time_meter),
				str(loss_meter),
				str(error_meter),
			]))

		return time_meter.value(), loss_meter.value(), error_meter.value()



	for epc in range(epochs):






		scaled_model = ModelWithTemperature(model)
		scaled_model.set_temperature(valid_loader)



		## Treino para cada epoca
	start_train_time = time.time()