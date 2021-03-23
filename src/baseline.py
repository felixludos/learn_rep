import sys, os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import omnifig as fig

import omnilearn as fd
from omnilearn import models
from omnilearn import util


@fig.Component('dislib-enc')
class Disentanglement_lib_Encoder(fd.Encodable, fd.Optimizable, fd.Function):
	def __init__(self, A, din=None, dout=None, **kwargs):
		
		if din is None:
			din = A.pull('in-shape', '<>din')
		if dout is None:
			dout = A.pull('latent-dim', '<>dout')

		C, H, W = din
		
		flat_shape = 64, H // 2**4, W // 2**4
		flat_dim = int(np.product(flat_shape))
		
		hidden_dim = A.pull('hidden_dim', 256)
		
		net = nn.Sequential(
			nn.Conv2d(C, 32, kernel_size=2, stride=2),
			util.get_nonlinearity(A.pull('nonlin', 'relu')),
			nn.Conv2d(32, 32, kernel_size=2, stride=2),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.Conv2d(32, 64, kernel_size=2, stride=2),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.Conv2d(64, 64, kernel_size=2, stride=2),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.Flatten(),
			nn.Linear(flat_dim, hidden_dim),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.Linear(hidden_dim, dout),
		)
		
		super().__init__(A, din=din, dout=dout, **kwargs)

		self.net = net

	def forward(self, x):
		return self.net(x)

	def encode(self, x):
		return self(x)


@fig.Component('dislib-dec')
class Disentanglement_lib_Decoder(fd.Decodable, fd.Optimizable, fd.Function):
	def __init__(self, A, din=None, dout=None, **kwargs):
		
		if din is None:
			din = A.pull('latent-dim', '<>din')
		if dout is None:
			dout = A.pull('out-shape', '<>dout')

		C, H, W = dout
		
		flat_shape = 64, H // 2**4, W // 2**4
		flat_dim = int(np.product(flat_shape))
		
		hidden_dim = A.pull('hidden_dim', 256)
		
		net = nn.Sequential(
			nn.Linear(din, hidden_dim),
			util.get_nonlinearity(A.pull('nonlin', 'relu')),
			nn.Linear(hidden_dim, flat_dim),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			models.Reshaper(flat_shape),
			nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.ConvTranspose2d(32, C, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity('sigmoid')
		)
		
		super().__init__(A, din=din, dout=dout, **kwargs)
		
		self.net = net

	def forward(self, q):
		return self.net(q)

	def decode(self, q):
		return self(q)

