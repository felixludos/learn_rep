import sys, os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import omnifig as fig

import foundation as fd
from foundation import models
from foundation import util


@fig.Component('ladder-enc')
class Ladder_Encoder(fd.Encodable, fd.Model):
	def __init__(self, A, din=None, dout=None, **kwargs):
		
		if 'latent_dim' in A:
			A.latent_dim = None
		
		layers = A.pull('layers')
		csizes = layers._conv_shapes[1:]
		din = layers.din
		rung_dims = A.pull('rung_dims')
		
		reverse_order = A.pull('reverse_order', False)
		
		layer_key = A.pull('layer_key', None)
		if layer_key is not None:
			layers = getattr(layers, layer_key, layers)
		
		try:
			len(rung_dims)
		except TypeError:
			rung_dims = [rung_dims ] *len(layers)
		
		ret_normal = isinstance(self, models.Normal_Distrib_Model)
		if ret_normal:
			rung_dims = [( 2 *r if r is not None else r) for r in rung_dims]
		
		latent_dim = sum(rung for rung in rung_dims if rung is not None)
		
		assert len(layers) == len(rung_dims)
		assert latent_dim > 0, 'no latent dim'
		
		super().__init__(A, din=din, dout=latent_dim, **kwargs)
		
		create_rung = A.pull('rungs')
		rungs = []
		
		for i, (rdim, cdin) in enumerate(zip(rung_dims, csizes)):
			
			if rdim is None:
				rungs.append(None)
			else:
				nxt = create_rung.current()
				nxt.din = cdin
				nxt.dout = rdim
				rung = next(create_rung)
				rungs.append(rung)
		
		while rungs[-1] is None:
			rungs.pop()
			layers.pop()
		
		self.ret_distrib = ret_normal
		self.rung_dims = rung_dims
		self.layers = nn.ModuleList(layers)
		self.rungs = nn.ModuleList(rungs)
		self.reverse_order = reverse_order
	
	def forward(self, x):
		
		qs = []
		
		B = x.size(0)
		
		c = x
		for l, r in zip(self.layers, self.rungs):
			c = l(c)
			if r is not None:
				q = r(c)
				qs.append(q.view(B ,2 ,-1) if self.ret_distrib else q)
		
		if self.reverse_order:
			qs = reversed(qs)
		
		q = torch.cat(qs ,-1)
		if self.ret_distrib:
			q = q.view(B, -1)
		return q

