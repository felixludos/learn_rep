


import sys, os, time, shutil#, traceback, ipdb
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

import numpy as np
#%matplotlib tk
import matplotlib.pyplot as plt

import foundation as fd
from foundation import models
from foundation import util
from foundation import train as trn
from foundation import data




@fd.Component('branched-dec')
class Branched(fd.Decodable, fd.Visualizable, fd.Schedulable, fd.Model):

	def __init__(self, A):

		latent_dim = A.pull('latent_dim', '<>din')
		out_shape = A.pull('out_shape', '<>dout')

		super().__init__(latent_dim, out_shape)

		root_dim = A.pull('root_dim', 0)
		splits = A.pull('splits')

		channels = A.pull('channels')

		factors = A.pull('factors', 2) # each conv layer
		try:
			len(factors)
		except TypeError:
			factors = [factors]
		if len(factors) != len(channels):
			factors = factors * len(channels)
			factors = factors[:len(channels)]


		total = int(np.product((factors)))

		C, H, W = out_shape

		in_shape = channels[0], H//total, W//total
		if in_shape[1] == 0 or in_shape[2] == 0:
			raise Exception(f'no features left: {in_shape} vs {factors}')

		if root_dim > 0:

			assert 'root' in A, 'no root network provided'
			A.root.din = root_dim
			A.root.dout = in_shape
			root = A.pull('root')

		else:
			root = nn.Parameter(torch.randn(*in_shape).unsqueeze(0), requires_grad=True)

		self.root = root
		self.root_dim = root_dim

		self.splits = (latent_dim - self.root_dim) // splits


	def _visualize(self, out, logger):
		pass


	def forward(self, q):

		if self.root_dim > 0:
			root, q = q[:, self.root_dim:], q[:, self.root_dim:]

			root = self.root(root)
		else:
			root = self.root

		branches = q.split(self.splits)

		pass




