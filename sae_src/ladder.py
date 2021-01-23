import sys, os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as distrib

import omnifig as fig

import foundation as fd
from foundation import models
from foundation import util

from .structure_modules import _find_dims


@fig.Component('rung-infer')
class InferenceRung(models.StyleExtractorLayer):
	def __init__(self, A, din=None, dout=None, style_dim=None, **kwargs):
		
		if style_dim is None:
			style_dim = A.pull('style-dim')
		
		din, dout = _find_dims(A, din, dout)
		
		net_info = A.pull('net', silent=True, raw=True)
		if net_info is not None:
			net_info.push('din', din, silent=True)
			net_info.push('latent-dim', style_dim, silent=True)
			net_info.push('dout', style_dim, silent=True)
		net = A.pull('net')
		
		super().__init__(A, din=din, dout=dout, style_dim=style_dim, **kwargs)
		
		self.net = net
		
	def extract(self, inp, **unused):
		return inp, self.net(inp)
	

@fig.Component('rung-gen')
class GenerativeRung(models.StyleFusionLayer):
	def __init__(self, A, din=None, dout=None, style_dim=None, feature_dim=None, **kwargs):
		
		if style_dim is None:
			style_dim = A.pull('style-dim')
		
		if feature_dim is None:
			feature_dim = A.pull('feature-dim', None)
		
		extra_dim = style_dim if feature_dim is None else feature_dim
		
		if din is None:
			din = A.pull('din', None)
			
		if din is not None:
			C, H, W = din
			dout = C+extra_dim, H, W
			
		else:
			
			if dout is None:
				dout = A.pull('dout', None)
		
			C, H, W = dout
			
			din = C-extra_dim, H, W
		
		super().__init__(A, din=din, style_dim=style_dim, dout=dout, **kwargs)
		
		net = None
		if feature_dim is not None:
			net_info = A.pull('net', None, silent=True, raw=True)
			if net_info is not None:
				net_info.push('din', style_dim, silent=True)
				net_info.push('latent-dim', style_dim, silent=True)
				net_info.push('dout', feature_dim, silent=True)
			net = A.pull('net')
		
		self.net = net
		self.extra_dim = extra_dim
	
	def process_style(self, style):
		style = super().process_style(style)
		if self.net is not None:
			style = self.net(style) # corresponds to network "v" in VLAE paper
		return style
	
	def infuse(self, content, style, **kwargs):
		
		B = style.size(0)
		
		style = style.unsqueeze(-1).unsqueeze(-1).expand(B, self.extra_dim, *content.shape[2:])
		
		return torch.cat([content, style], 1)  # concatenates the noise


@fig.Component('ladder-enc')
class LadderEncoder(fd.Encodable, models.StyleExtractor):
	def __init__(self, A, ret_style=None, ret_content=None, reverse_order=True, **kwargs):
		
		reverse_order = A.pull('reverse_order', reverse_order)
		
		super().__init__(A, ret_style=True, ret_content=False, **kwargs)
		
		if reverse_order:
			self.style_layers = self.style_layers[::-1] # first layer is the "top" rung (deepest)
		self.reverse_order = reverse_order
	
	def collect_style(self):
		
		style = super().collect_style()
		
		if isinstance(self, models.Normal):
			B = style.size(0)
			style = style.view(B, -1, 2).permute(0,2,1).contiguous().view(B,-1)
			
		return style