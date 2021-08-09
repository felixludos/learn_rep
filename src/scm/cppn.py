
import sys, os

import torch

from omnibelt import unspecified_argument

import omnifig as fig

import omnilearn as learn
from omnilearn import util


@fig.Component('cppn')
class CPPN(util.Seeded, learn.Function):
	def __init__(self, A, feature_dim=None, size=unspecified_argument,
	             include_radial=None, include_neg_radial=None, grid_dim=None,
	             net=None, net_config=None, net_hidden=unspecified_argument, init_normal=None,
	             grid=None, din=None, dout=None,
	             **kwargs):
		
		if feature_dim is None:
			feature_dim = A.pull('feature-dim')
		
		if include_radial is None:
			include_radial = A.pull('include-radial', False)
		if include_neg_radial is None:
			include_neg_radial = A.pull('include-neg-radial', True)
		
		if size is unspecified_argument:
			size = A.pull('out-shape', '<>dout', None)
			
		if size is None:
			H = A.pull('height', '<>H', 64)
			W = A.pull('width', '<>W', 64)
			size = H, W
		
		if len(size) == 2:
			C = A.pull('channels', '<>C')
			size = (C, *size)
		assert len(size) == 3, f'bad size: {size}'
		
		C, H, W  = size
		
		if din is None:
			din = feature_dim
		if dout is None:
			dout = size
		
		if grid is None:
			grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)))
			
			if include_radial:
				grid = torch.cat([grid, grid.pow(2).sum(0, keepdim=True).sqrt()])
			if include_neg_radial:
				grid = torch.cat([grid, 1 - grid.pow(2).sum(0, keepdim=True).sqrt()])
		
		if grid_dim is None:
			grid_dim = grid.size(0)
		
		if net is None and net_config is None:
			A.push('net._type', 'multilayer', overwrite=False, silent=True)
			A.push('net.nonlin', 'tanh', overwrite=False, silent=True)
			A.push('net.kernel', 1, overwrite=False, silent=True)
			A.push('net.conv_layer', 'conv-layer', overwrite=False, silent=True)
			A.push('net.last._type', '<>conv_layer', overwrite=False, silent=True)
			# A.push('conv_kwargs.bias', False)
			A.push('net.last.nonlin', 'sigmoid', overwrite=False, silent=True)
			
			net_config = A.pull('net', raw=True)
			
		if net_hidden is unspecified_argument:
			net_hidden = A.pull('net-hidden', '<>hidden', None)
		if net_hidden is not None and len(net_hidden):
			net_config.push('layers', [{'channels': c} for c in net_hidden], overwrite=False, silent=True)
			for l in net_config.pull('layers', raw=True):
				l.push('_type', '<>conv_layer', overwrite=False, silent=True)
		
		net_dim = feature_dim + grid_dim, H, W
		
		if net is None:
			net_config.push('din', net_dim, silent=True)
			net_config.push('dout', dout, silent=True)
		
		net = net_config.pull_self()
		
		if init_normal is None:
			init_normal = A.pull('init-normal', True)
		
		super().__init__(A, din=din, dout=dout, **kwargs)

		self.init_normal = init_normal

		self.register_buffer('grid', grid)
		
		self.feature_dim = feature_dim
		self.net_dim = net_dim
		
		self.net = net
		self._initialize_net(A)
		
	def _initialize_net(self, A):
		if self.init_normal:
			for param in self.net.parameters():
				param.data.normal_(generator=self.gen)
		
	def _expand_grid(self, x):
		return self.grid.unsqueeze(0).expand(x.size(0), *self.grid.size())
	
	def _expand_features(self, x):
		return x.unsqueeze(-1).unsqueeze(-1).expand(*x.size(), *self.grid.size()[1:])
		
	def forward(self, x):
		feats = torch.cat([self._expand_grid(x), self._expand_features(x)], 1)
		return self.net(feats)






