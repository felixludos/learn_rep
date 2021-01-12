#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn

import omnifig as fig

import numpy as np
#%matplotlib tk

import foundation as fd
from foundation import models

from .structure_modules import StyleLayer

@fig.Component('style-dec')
class StyleDecorder(fd.Decodable, models.MultiLayer):

	def __init__(self, A, din=None, dout=None, **kwargs):

		split_style = A.pull('split-style', False)

		super().__init__(A, din=din, dout=dout, in_order=False, **kwargs)

		din, dout = self.get_dims()

		self.root = nn.Parameter(torch.randn(*din), requires_grad=True)

		self.style_layers = [layer for layer in self.layers if isinstance(layer, StyleLayer)]

		style_dims = None
		if split_style:
			style_dims = [layer.style_dim for layer in self.style_layers]
			din = sum(style_dims)
		else:
			din = self.style_layers[0].style_dim
			for layer in self.style_layers:
				assert layer.style_dim == din

		self.din = din
		self.style_dims = style_dims

	def forward(self, q):

		# cache styles
		styles = [q] * len(self.style_layers) if self.style_dims is None else q.split(self.style_dims, dim=1)
		for layer, style in zip(self.style_layers, styles):
			layer.cache_style(style)

		# run forward pass through all layers starting with "root"
		return super().forward(self.root.unsqueeze(0).expand(q.size(0), *self.root.shape))

