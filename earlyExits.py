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


class EarlyExitDNN(nn.Module):

	def __init__(self, modelName, n_branchs, position_list, n_classes, input_dim, device, pretrained=True):

		super(B_AlexNet, self).__init__()

		self.modelName = modelName
		self.n_branchs = n_branchs
		self.position_list = position_list
		self.n_classes = n_classes
		self.input_dim = input_dim
		self.device = device
		self.pretrained = pretrained

		build_early_exit_dnn = self.dnn_architecture_model()
	    build_early_exit_dnn()


	def dnn_architecture_model(self):

		"""
	    This method selects the backbone to insert the early exits.
	    """

	    architecture_dnn_model_dict = {"alenet": self.early_exit_alexnet,
	    							   "mobilenet": self.early_exit_mobilenet,
	                                   "efficientnet_b1": self.early_exit_efficientnet_b1}

	    self.pool_size = 7 if (self.model_name == "vgg16") else 1   				#### ver pq 7 ou 1
	    return architecture_dnn_model_dict.get(self.model_name, self.invalid_model)


	def early_exit_block(self):



	def early_exit_alexnet(self):
		"""
		This method inserts early exits into a Alexnet model
		"""

		self.stages = nn.ModuleList()
		self.exits = nn.ModuleList()
		self.layers = nn.ModuleList()
		self.cost = []
		self.stage_id = 0

		# Loads the backbone model. In other words, Alexnet architecture provided by Pytorch.
		backbone_model = models.alexnet(self.pretrained)

		# It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
		self.verifies_nr_exit_alexnet(backbone_model.features)

		# This obtains the flops total of the backbone model
		self.total_flops = self.countFlops(backbone_model)

		# This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
		self.threshold_flop_list = self.where_insert_early_exits()

		for layer in backbone_model.features:
		  self.layers.append(layer)
		  if (isinstance(layer, nn.ReLU)) and (self.is_suitable_for_exit()):
		    self.add_exit_block()



		self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6, 6)))
		self.stages.append(nn.Sequential(*self.layers))


		self.classifier = backbone_model.classifier
		self.classifier[6] = nn.Linear(in_features=4096, out_features=self.n_classes, bias=True)
		self.softmax = nn.Softmax(dim=1)
		self.set_device()










