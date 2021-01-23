#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from torch import distributions as distrib

import omnifig as fig

import numpy as np
#%matplotlib tk

import foundation as fd
from foundation import models

@fig.Component('style-dec')
class StyleDecoder(fd.Decodable, models.StyleFusion):

	def __init__(self, A, in_order=False, **kwargs):
		
		in_order = A.pull('in_order', in_order)

		super().__init__(A, in_order=in_order, **kwargs)

		din, dout = self.get_dims()

		self.root = nn.Parameter(torch.randn(*din), requires_grad=True)
		self.din = self.get_style_dim()

	def get_content(self, N):
		return self.root.unsqueeze(0).expand(N, *self.root.shape)

	def forward(self, q):
		return super().forward(self.get_content(q.size(0)), q)













