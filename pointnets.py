


import sys, os, time, shutil#, traceback, ipdb
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import configargparse

import numpy as np
#%matplotlib tk
import matplotlib.pyplot as plt

import umap, shap
import umap.plot
import gpumap
from sklearn.decomposition import PCA

import foundation as fd
from foundation import models
from foundation import util
from foundation import train as trn
from foundation import data


@fd.Component('patch-enc')
class PointEncoder(fd.Encodable, fd.Visualizable, fd.Schedulable, fd.Model):

	def __init__(self, A):

		in_shape = A.pull('in_shape', '<>din')
		latent_dim = A.pull('latent_dim', '<>dout')

		assert 'transform' in A, 'transofrm required to go from image to point cloud'
		A.transform.in_shape = in_shape

		transform = A.pull('transform') # converts the image into a point cloud

		assert 'pointnet' in A, 'pointnet required for this encoder'
		A.pointnet.pin = transform.pout
		
		pointnet = A.pull('pointnet')

		super().__init__(in_shape, latent_dim)

		self.transform = transform
		self.pointnet = pointnet

		self.set_optim(A)
		self.set_schedule(A)

	def _visualize(self, out, logger): # TODO
		pass

	def forward(self, x): # images

		p = self.transform(x)
		q = self.pointnet(p)

		return q


@fd.AutoComponent('patch-points')
class Patch_Points(fd.Model):
	'''
	Converts an image into a set of unordered points where each point is a concat of the pixels of
	a unique patch of the image.
	'''
	def __init__(self, in_shape, patch_size=4, stride=2, dilation=1, padding=0):

		C, H, W = in_shape

		try:
			len(patch_size)
		except TypeError:
			patch_size = (patch_size, patch_size)

		try:
			len(stride)
		except TypeError:
			stride = (stride, stride)

		try:
			len(dilation)
		except TypeError:
			dilation = (dilation, dilation)

		try:
			len(padding)
		except TypeError:
			padding = (padding, padding)

		pout = C*patch_size[0]*patch_size[1]

		Nh = (H + 2 * padding[0] - dilation[0] * (patch_size[0] - 1) - 1) // stride[0] + 1
		Nw = (W + 2 * padding[1] - dilation[1] * (patch_size[1] - 1) - 1) // stride[1] + 1
		N = Nh * Nw

		super().__init__(in_shape, (pout, N))

		self.pout = pout

		self.transform = nn.Unfold(kernel_size=patch_size, stride=stride,
		                           dilation=dilation, padding=padding)

	def forward(self, x):
		return self.transform(x)



@fd.AutoComponent('1dconv-net')
def make_pointnet(pin, pout, hidden_dims=[],
                  nonlin='prelu', output_nonlin=None):
	nonlins = [nonlin] * len(hidden_dims) + [output_nonlin]
	hidden_dims = [pin] + hidden_dims + [pout]

	layers = []
	for in_dim, out_dim, nonlin in zip(hidden_dims, hidden_dims[1:], nonlins):
		layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1))
		if nonlin is not None:
			layers.append(util.get_nonlinearity(nonlin))

	net = nn.Sequential(*layers)
	net.pin = pin
	net.pout = pout
	return net


@fd.Component('pointnet')
class PointNet(fd.Trainable_Model):
	def __init__(self, A):

		pin = A.pull('pin') # should be the number of channels of the points
		dout = A.pull('latent_dim', '<>dout')

		create_module = A.pull('modules')

		final = A.pull('final', None)
		pool = A.pull('pool', None)

		super().__init__(din=pin, dout=dout)

		modules = []


		nxt = create_module.current()
		assert nxt is not None, 'no point-net modules provided'
		nxt.pin = pin

		for module in create_module:

			modules.append(module)

			nxt = create_module.current()
			if nxt is not None:
				nxt.pin = module.pout
				nxt.pin1 = module.pout1
				nxt.pin2 = module.pout2
				if nxt._type == 'point-dual':
					if 'left' in nxt:
						nxt.left.pin = module.pout1
					if 'right' in nxt:
						nxt.right.pin = module.pout2

		pout = module.pout
		assert pout is not None, f'last module must have a single output: {module}'

		self.modules = nn.Sequential(*modules)
		self.pool = pool

		if final is not None:
			assert final.din == pout and final.dout == dout, f'invalid {final.din} v {pout} and {final.dout} v {dout}'
		elif pout != dout: # by default fix output to correctly sized output using a dense layer
			final = nn.Linear(pout, dout)
		self.final = final

	def forward(self, pts):

		pts = self.modules(pts)

		B, C, *rest = pts.size()
		if len(rest):
			if rest[0] > 1:
				pts = self.pool(pts)
			pts = pts.view(B,C)
		out = pts

		if self.final is not None:
			out = self.final(out)

		return out

# Abstract modules

class PointNetModule(fd.Model):
	def __init__(self, pin=None, pout=None,
	             pin1=None, pout1=None, pin2=None, pout2=None):
		super().__init__(pin, pout)
		self.pin = pin
		self.pout = pout

		self.pin1 = pin1
		self.pout1 = pout1
		self.pin2 = pin2
		self.pout2 = pout2



class PointSplit(PointNetModule):
	def __init__(self, pin, pout1, pout2):
		super().__init__(pin=pin, pout1=pout1, pout2=pout2)

class PointTransform(PointNetModule):
	def __init__(self, pin, pout):
		super().__init__(pin=pin, pout=pout)

class PointParallel(PointNetModule):
	def __init__(self, pin1, pin2, pout1, pout2):
		super().__init__(pin1=pin1, pin2=pin2, pout1=pout1, pout2=pout2)

	def __call__(self, pts): # so the point net can connect modules by nn.Sequential
		return super().__call__(*pts)

class PointJoin(PointNetModule):
	def __init__(self, pin1, pin2, pout):
		super().__init__(pin1=pin1, pin2=pin2, pout=pout)

	def __call__(self, pts): # so the point net can connect modules by nn.Sequential
		return super().__call__(*pts)


# Point-net operations


@fd.AutoComponent('point-split')
class PointSplitter(PointSplit):
	def __init__(self, pin, split):
		super().__init__(pin=pin, pout1=split, pout2=pin-split)

		assert pin > split, f'not enough dimensions to split: {pin} vs {split}'
		self.split = split

	def forward(self, p):
		return p[:,:self.split], p[:,self.split:]


@fd.AutoComponent('point-transform')
class PointSelfTransform(PointTransform):
	def __init__(self, net):
		super().__init__(net.pin, net.pout)

		self.net = net

	def forward(self, p):
		return self.net(p)


@fd.AutoComponent('point-dual')
class PointDualTransform(PointParallel):
	def __init__(self, left=None, right=None, pin1=None, pin2=None):

		assert left is not None or pin1 is not None
		assert right is not None or pin2 is not None

		if left is not None:
			pin1 = left.pin
			pout1 = left.pout
		else:
			pout1 = pin1

		if right is not None:
			pin2 = right.pin
			pout2 = right.pout
		else:
			pout2 = pin2

		super().__init__(pin1, pin2, pout1, pout2)

		self.left = left
		self.right = right

	def forward(self, p1, p2):
		if self.left is not None:
			p1 = self.left(p1)
		if self.right is not None:
			p2 = self.right(p2)
		return p1, p2

@fd.AutoComponent('point-swap')
class PointSwap(PointParallel):
	def __init__(self, pin1, pin2):
		super().__init__(pin1=pin1, pin2=pin2, pout1=pin2, pout2=pin1)

	def forward(self, p1, p2):
		return p2, p1


@fd.AutoComponent('point-cat')
class PointJoiner(PointJoin):
	def __init__(self, pin1, pin2):
		super().__init__(pin1=pin1, pin2=pin2, pout=pin1+pin2)

	def forward(self, p1, p2): # TODO make sure shapes work out
		return torch.cat([p1, p2], dim=1)




