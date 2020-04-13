


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




@fd.Component('branch-dec')
class Branched_Decoder(fd.Decodable, fd.Visualizable, fd.Schedulable, fd.Model):

	def __init__(self, A):

		latent_dim = A.pull('latent_dim', '<>din')
		out_shape = A.pull('out_shape', '<>dout')

		super().__init__(latent_dim, out_shape)

		root_dim = A.pull('root_dim', 0)
		branch_dim = A.pull('branch_dim')

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

		try:
			len(branch_dim)
		except TypeError:
			branch_dim = [branch_dim]
		if len(branch_dim) != len(channels):
			branch_dim = branch_dim * len(channels)
			branch_dim = branch_dim[:len(channels)]

		assert latent_dim == root_dim + sum(branch_dim), f'not the right split: {latent_dim} vs {root_dim} {branch_dim}'

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
			root = nn.Parameter(torch.randn(*in_shape), requires_grad=True)

		self.root = root
		self.root_dim = root_dim

		self.branch_dim = branch_dim

		create_branch = A.pull('branches')
		create_layer = A.pull('layers')

		channels = list(channels) + [C]

		bdims = iter(branch_dim)

		branches = []
		layers = []

		din = in_shape
		for i, (in_chn, out_chn, fct) in enumerate(zip(channels, channels[1:], factors)):

			nxt = create_branch.current()
			nxt.din = next(bdims)
			nxt.features = in_chn

			branches.append(next(create_branch))

			nxt = create_layer.current()
			nxt.din = din
			nxt.in_channels = in_chn
			nxt.out_channels = out_chn
			nxt.factor = fct

			layer = next(create_layer)
			layers.append(layer)

			din = layer.dout

		assert din == out_shape, f'error: {din} vs {out_shape}'

		self.branches = nn.ModuleList(branches)
		self.layers = nn.ModuleList(layers)

		assert len(self.branches) == len(self.branch_dim) == len(self.layers)


	def _visualize(self, out, logger):
		pass


	def forward(self, q):

		if self.root_dim > 0:
			root, q = q[:, :self.root_dim], q[:, self.root_dim:]
			c = self.root(root)
		else:
			B = q.size(0)
			c = self.root
			shape = c.shape
			c = c.expand(B, *shape)

		styles = q.split(self.branch_dim, dim=1)

		for branch, style, conv in zip(self.branches, styles, self.layers):

			c = branch(c, style)
			c = conv(c)

		return c




