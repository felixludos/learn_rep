
import sys, os, time, shutil#, traceback, ipdb
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

import numpy as np
#%matplotlib tk
import matplotlib.pyplot as plt
from matplotlib import cm

import foundation as fd
from foundation import models
from foundation import util
from foundation import train as trn
from foundation import data


# @fd.AutoModifier('demeaned')
class Demeaned(fd.Model):
	pass

class Invertible_Fully_Connected(fd.Model):
	def __init__(self, dim, nonlin):
		pass

class PCA(fd.Model):
	pass

