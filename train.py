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
from temperature_scaling import ModelWithTemperature



if __name__ == '__main__':

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	epoch = 100

	## Tx de apredizado
	lr = 0.001 ##ou 
	#lr=1.5e-4

	#Otimazador
	opt = 'SGD'
	# opt = 'adam'

	dataset = 'cifar10'

	bt_size = 128

	classes_list, label_list,train_loader, valid_loader = tools.data_set(dataset,bt_size,train=True)

	model = ee.EarlyExitDNN()
	model = model.to(device)


	# Paremetros de configuracao da rede neural
	criterion, optimize = tools.parameter(model,lr,opt)
	criterion = criterion.to(device)
	softmax = nn.Softmax(dim=1)


	# Preparar o que eu quero de resutados






	## chamar a rede para treinar
	for epc in range(epochs):

	    model.train()

	    for images,target in tqdm(train_loader):

	        images, target = images.to(device), target.to(device)


	scaled_model = ModelWithTemperature(model)
	scaled_model.set_temperature(valid_loader)



		## Treino para cada epoca
		start_train_time = time.time()