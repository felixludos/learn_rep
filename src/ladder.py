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

from .structure_modules import _find_dims


@fig.Component('rung')
class LadderRung(models.StyleExtractor):
	def __init__(self, A, din=None, dout=None, style_dim=None, **kwargs):
		
		if style_dim is None:
			style_dim = A.pull('style-dim')
		
		din, dout = _find_dims(A, din, dout)
		feature_dim = din[0] if isinstance(din, (tuple, list)) else din
		
		net_info = A.pull('net', None, silent=True, raw=True)
		if net_info is not None:
			net_info.push('din', style_dim, silent=True)
			net_info.push('latent-dim', style_dim, silent=True)
			net_info.push('dout', feature_dim, silent=True)
		net = A.pull('net', None)
		
		C, H, W = din
		
		extra_dim = style_dim if net is None else feature_dim
		
		dout = C, H, W
		
		super().__init__(A, din=din, style_dim=style_dim, dout=dout, **kwargs)
		
		self.extra_dim = extra_dim
		self.net = net
	
	def process_style(self, style):
		style = super().process_style(style)
		if self.net is not None:
			style = self.net(style)
		return style
	
	def infuse(self, content, style, **kwargs):
		
		B = style.size(0)
		
		style = style.unsqueeze(-1).unsqueeze(-1).expand(B, self.extra_dim, *content.shape[2:])
		
		return torch.cat([content, style], 1)  # concatenates the noise



@fig.Component('ladder-enc')
class LadderEncoder(fd.Encodable, models.StyleCollector):
	def __init__(self, A, **kwargs):
		
		reverse_order = A.pull('reverse_order', False)
		
		super().__init__(A, **kwargs)
		
		self.reverse_order = reverse_order
	
	
