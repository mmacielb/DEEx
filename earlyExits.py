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


	def dnn_model():
		








	