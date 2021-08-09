import sys, os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch import distributions as distrib
from torch.nn import functional as F

from omnibelt import unspecified_argument

import omnifig as fig

import omnilearn as learn
from omnilearn import models
from omnilearn import util



@fig.Component('dislib-enc-arch')
class Disentanglement_lib_Encoder_Arch(learn.Encodable, learn.Optimizable, learn.Function):
	def __init__(self, A, din=None, dout=None, hidden_dim=None, ret_normal=None, **kwargs):
		
		if din is None:
			din = A.pull('in-shape', '<>din')
		if dout is None:
			dout = A.pull('latent-dim', '<>dout')
		
		if ret_normal is None:
			ret_normal = A.pull('ret-normal', True)
		
		latent_dim = dout // 2
		
		C, H, W = din
		
		flat_shape = 64, H // 2 ** 4, W // 2 ** 4
		flat_dim = int(np.product(flat_shape))
		
		if hidden_dim is None:
			hidden_dim = A.pull('hidden_dim', 256)
		
		conv = nn.Sequential(
			nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity(A.pull('nonlin', 'relu')),
			nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.Conv2d(32, 64, kernel_size=2, stride=2),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.Conv2d(64, 64, kernel_size=2, stride=2),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			# nn.Flatten(),
		)
		fc = nn.Sequential(
			nn.Linear(flat_dim, hidden_dim),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
		)
		
		super().__init__(A, din=din, dout=dout, **kwargs)
		
		self.conv = conv
		self.fc = fc
		
		self.mu_layer = nn.Linear(hidden_dim, latent_dim)
		self.logsigma_layer = nn.Linear(hidden_dim, latent_dim)
		
		self.ret_normal = ret_normal and not isinstance(self, models.Normal)
	
	def forward(self, x):
		c = self.conv(x)
		c = c.permute(0,2,3,1).reshape(c.size(0), -1)
		c = self.fc(c)
		
		mu = self.mu_layer(c)
		logsigma = self.logsigma_layer(c)
		
		if self.ret_normal:
			return distrib.Normal(mu, logsigma.exp())
		return torch.cat([mu, logsigma], -1)
		
	
	def encode(self, x):
		return self(x)
	
@fig.Component('dislib-enc')
class Disentanglement_lib_Encoder(learn.TensorflowPort, Disentanglement_lib_Encoder_Arch):
	def __init__(self, A, tf_path=unspecified_argument, din=None, dout=None, **kwargs):
		
		if tf_path is unspecified_argument:
			tf_path = self._process_tf_path(A, tf_path=tf_path)
		if tf_path is not None:
			din, dout = (3, 64, 64), 20
	
		super().__init__(A, tf_path=tf_path, din=din, dout=dout, **kwargs)
	
	# def _convert_tf_val(self, my_name, param, tf_val):
	# 	if my_name == 'fc.0.weight':
	# 		val = torch.from_numpy(tf_val).float()
	# 		val = val.reshape(64, 4, 4, 256).permute(3,1,2,0).reshape(256, 1024)
	# 		# val = val.reshape(256, 64, 4, 4).permute(0,3,2,1).reshape(val.size(0),-1)
	# 		# val = val.reshape(256, 64, 4, 4).permute(0,2,3,1).reshape(val.size(0),-1)
	# 		# val = val.reshape(256, 4, 4, 64).permute(0,2,3,1).reshape(val.size(0),-1)
	# 		return val
	#
	# 	val = super()._convert_tf_val(my_name, param, tf_val)
	#
	# 	return val

	_torch_save_name = 'encoder'
	_tf_var_names = {'conv.0.weight': 'encoder/e1/kernel',
	              'conv.0.bias': 'encoder/e1/bias',
	              'conv.2.weight': 'encoder/e2/kernel',
	              'conv.2.bias': 'encoder/e2/bias',
	              'conv.4.weight': 'encoder/e3/kernel',
	              'conv.4.bias': 'encoder/e3/bias',
	              'conv.6.weight': 'encoder/e4/kernel',
	              'conv.6.bias': 'encoder/e4/bias',
	              'fc.0.weight': 'encoder/e5/kernel',
	              'fc.0.bias': 'encoder/e5/bias',
	              # 'conv.9.weight': 'encoder/e5/kernel',
	              # 'conv.9.bias': 'encoder/e5/bias',
	              'mu_layer.weight': 'encoder/means/kernel',
	                 'mu_layer.bias': 'encoder/means/bias',
	                 'logsigma_layer.weight': 'encoder/log_var/kernel',
	                 'logsigma_layer.bias': 'encoder/log_var/bias',
	                 }



@fig.Component('dislib-dec-arch')
class Disentanglement_lib_Decoder_Arch(learn.Decodable, learn.Optimizable, learn.Function):
	def __init__(self, A, din=None, dout=None, **kwargs):
		
		if din is None:
			din = A.pull('latent-dim', '<>din')
		if dout is None:
			dout = A.pull('out-shape', '<>dout')
		
		C, H, W = dout
		
		flat_shape = 64, H // 2 ** 4, W // 2 ** 4
		flat_dim = int(np.product(flat_shape))
		
		hidden_dim = A.pull('hidden_dim', 256)
		
		fc = nn.Sequential(
			nn.Linear(din, hidden_dim),
			util.get_nonlinearity(A.pull('nonlin', 'relu')),
			nn.Linear(hidden_dim, flat_dim),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
		)
		
		conv = nn.Sequential(
			# models.Reshaper(flat_shape),
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
		
		self.flat_shape = flat_shape
		
		self.fc = fc
		self.conv = conv
	
	def forward(self, q):
		if isinstance(q, distrib.Normal):
			q = q.rsample()
			
		c = self.fc(q)
		C, H, W = self.flat_shape
		c = c.reshape(-1, H, W, C).permute(0, 3, 1, 2)
		
		return self.conv(c)
	
	def decode(self, q):
		return self(q)

@fig.Component('dislib-dec')
class Disentanglement_lib_Decoder(learn.TensorflowPort, Disentanglement_lib_Decoder_Arch):
	def __init__(self, A, tf_path=unspecified_argument, din=None, dout=None, **kwargs):
		
		if tf_path is unspecified_argument:
			tf_path = self._process_tf_path(A, tf_path=tf_path)
		if tf_path is not None:
			din, dout = 10, (3, 64, 64)
		
		super().__init__(A, tf_path=tf_path, din=din, dout=dout, **kwargs)
	
	# def _convert_tf_val(self, my_name, param, tf_val):
	# 	val = super()._convert_tf_val(my_name, param, tf_val)
	#
	# 	# if my_name == 'conv.0.weight':
	# 	# 	val = val.reshape(256, 4, 4, 64).permute(0, 2, 3, 1).reshape(val.size(0), -1)
	#
	# 	return val
	
	_torch_save_name = 'decoder'
	_tf_var_names = {
		'fc.0.weight': 'decoder/dense/kernel',
		'fc.0.bias': 'decoder/dense/bias',
		'fc.2.weight': 'decoder/dense_1/kernel',
		'fc.2.bias': 'decoder/dense_1/bias',
		
		'conv.0.weight': 'decoder/conv2d_transpose/kernel',
		'conv.0.bias': 'decoder/conv2d_transpose/bias',
		'conv.2.weight': 'decoder/conv2d_transpose_1/kernel',
		'conv.2.bias': 'decoder/conv2d_transpose_1/bias',
		'conv.4.weight': 'decoder/conv2d_transpose_2/kernel',
		'conv.4.bias': 'decoder/conv2d_transpose_2/bias',
		'conv.6.weight': 'decoder/conv2d_transpose_3/kernel',
		'conv.6.bias': 'decoder/conv2d_transpose_3/bias'
		
		# 'conv.1.weight': 'decoder/conv2d_transpose/kernel',
		# 'conv.1.bias': 'decoder/conv2d_transpose/bias',
		# 'conv.3.weight': 'decoder/conv2d_transpose_1/kernel',
		# 'conv.3.bias': 'decoder/conv2d_transpose_1/bias',
		# 'conv.5.weight': 'decoder/conv2d_transpose_2/kernel',
		# 'conv.5.bias': 'decoder/conv2d_transpose_2/bias',
		# 'conv.7.weight': 'decoder/conv2d_transpose_3/kernel',
		# 'conv.7.bias': 'decoder/conv2d_transpose_3/bias'
	}



# class Disentanglement_lib_Encoder(learn.Encodable, learn.Optimizable, learn.Function):
# 	def __init__(self, A, din=None, dout=None, **kwargs):
#
# 		if din is None:
# 			din = A.pull('in-shape', '<>din')
# 		if dout is None:
# 			dout = A.pull('latent-dim', '<>dout')
#
# 		C, H, W = din
#
# 		flat_shape = 64, H // 2**4, W // 2**4
# 		flat_dim = int(np.product(flat_shape))
#
# 		hidden_dim = A.pull('hidden_dim', 256)
#
# 		net = nn.Sequential(
# 			nn.Conv2d(C, 32, kernel_size=4, stride=2),
# 			util.get_nonlinearity(A.pull('nonlin', 'relu')),
# 			nn.Conv2d(32, 32, kernel_size=4, stride=2),
# 			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
# 			nn.Conv2d(32, 64, kernel_size=2, stride=2),
# 			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
# 			nn.Conv2d(64, 64, kernel_size=2, stride=2),
# 			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
# 			nn.Flatten(),
# 			nn.Linear(flat_dim, hidden_dim),
# 			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
# 			nn.Linear(hidden_dim, dout),
# 		)
#
# 		super().__init__(A, din=din, dout=dout, **kwargs)
#
# 		self.net = net
#
# 	def forward(self, x):
# 		return self.net(x)
#
# 	def encode(self, x):
# 		return self(x)


# class Disentanglement_lib_Decoder(learn.Decodable, learn.Optimizable, learn.Function):
# 	def __init__(self, A, din=None, dout=None, **kwargs):
#
# 		if din is None:
# 			din = A.pull('latent-dim', '<>din')
# 		if dout is None:
# 			dout = A.pull('out-shape', '<>dout')
#
# 		C, H, W = dout
#
# 		flat_shape = 64, H // 2**4, W // 2**4
# 		flat_dim = int(np.product(flat_shape))
#
# 		hidden_dim = A.pull('hidden_dim', 256)
#
# 		net = nn.Sequential(
# 			nn.Linear(din, hidden_dim),
# 			util.get_nonlinearity(A.pull('nonlin', 'relu')),
# 			nn.Linear(hidden_dim, flat_dim),
# 			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
# 			models.Reshaper(flat_shape),
# 			nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
# 			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
# 			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
# 			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
# 			nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
# 			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
# 			nn.ConvTranspose2d(32, C, kernel_size=4, stride=2, padding=1),
# 			util.get_nonlinearity('sigmoid')
# 		)
#
# 		super().__init__(A, din=din, dout=dout, **kwargs)
#
# 		self.net = net
#
# 	def forward(self, q):
# 		return self.net(q)
#
# 	def decode(self, q):
# 		return self(q)

