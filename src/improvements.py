
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from omnibelt import unspecified_argument
import omnifig as fig

import omnilearn as learn
from omnilearn import util



class ImprovedBase(learn.FunctionBase):
	def __init__(self, model, space='base'):
		super().__init__(din=model.din, dout=model.dout, device=model.device)
		self.base_dim = model.decoder.din
		self.base = model # should be an autoencoder
		self.modifications = []
		self.named_mods = {}
		self.change_space(space)
	
	def change_space(self, space):
		assert space in {'base', 'clean', 'compact', 'split'}
		self.space = space
	
	def add_improvement(self, mod, name=None):
		if name is not None:
			self.named_mods[name] = mod
		self.modifications.append(mod)
	
	def modified_dims(self):
		dims = set()
		for mod in self.modifications:
			dims.update(mod.dims)
		return dims
	
	def transform(self, q, to=None, init='base'):
		raise NotImplementedError
	
	def sample_prior(self, N=1):
		
		
		
		return torch.randn(N, self.base_dim, device=self.device)

	


class Improvement(learn.FunctionBase):
	def __init__(self, dims, din=2, dout=2, **kwargs):
		super().__init__(din=din, dout=dout, **kwargs)
		self.dims = dims
		
	def transform_prior(self, x):
		return x
	
	def transform(self, x):
		return x


class MapSampling2D(Improvement):
	def __init__(self, dim, probs=None, logits=None, temperature=1., **kwargs):
		super().__init__(dims=[dim], **kwargs)
		assert probs is not None or logits is not None
		if probs is None:
			raise NotImplementedError
		
		






