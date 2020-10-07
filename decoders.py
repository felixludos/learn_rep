#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn

import omnifig as fig

import numpy as np
#%matplotlib tk

import foundation as fd
from foundation import legacy as lgcy



@fig.Component('branch-dec')
class Branched_Decoder(fd.Decodable, fd.Visualizable, fd.Schedulable, fd.Model):

	def __init__(self, A):

		latent_dim = A.pull('latent_dim', '<>din', None)
		out_shape = A.pull('out_shape', '<>dout')

		root_dim = A.pull('root_dim', 0)
		branch_dim = A.pull('branch_dim', None)
		split_latent = branch_dim is not None

		if latent_dim is None:
			assert branch_dim is not None
			latent_dim = root_dim + sum(br for br in branch_dim if br is not None)
			A.push('latent_dim', latent_dim)

		full_latent = latent_dim - root_dim

		super().__init__(latent_dim, out_shape)

		channels = A.pull('channels')


		factors = A.pull('factors', 2) # each conv layer
		try:
			len(factors)
		except TypeError:
			factors = [factors]
		if len(factors) != len(channels):
			factors = factors * len(channels)
			factors = factors[:len(channels)]

		total = int(np.product((factors)))

		try:
			len(branch_dim)
		except TypeError:
			branch_dim = [branch_dim]
		if len(branch_dim) != len(channels):
			branch_dim = branch_dim * len(channels)
			branch_dim = branch_dim[:len(channels)]

		assert not split_latent or latent_dim == root_dim + sum(br for br in branch_dim if br is not None), \
			f'not the right split: {latent_dim} vs {root_dim} {branch_dim}'

		C, H, W = out_shape

		in_shape = channels[0], H//total, W//total
		if in_shape[1] == 0 or in_shape[2] == 0:
			raise Exception(f'no features left: {in_shape} vs {factors}')

		if root_dim > 0:
			rtype = A.pull('root._type', None, silent=True)
			assert rtype is not None, 'no root network provided'
			rA = A['root']
			rA.push('din', root_dim, silent=True)
			rA.push('dout', in_shape, silent=True)
			
			root = A.pull('root')

		else:
			root = nn.Parameter(torch.randn(*in_shape), requires_grad=True)

		self.root = root
		self.root_dim = root_dim

		self.split_latent = split_latent
		self.branch_dim = [br for br in branch_dim if br is not None]

		create_branch = A.pull('branches')
		create_layer = A.pull('layers')

		channels = list(channels) + [C]

		bdims = iter(branch_dim)

		branches = []
		layers = []

		din = in_shape
		for i, (in_chn, out_chn, fct) in enumerate(zip(channels, channels[1:], factors)):

			bdim = next(bdims)
			if split_latent and bdim is None:
				branches.append(None)
			else:
				if bdim is None:
					bdim = full_latent
				nxt = create_branch.view()
				nxt.push('din', bdim, silent=True)
				nxt.push('features', in_chn, silent=True)
	
				branch = next(create_branch)
				branches.append(branch)
				din = branch.dout, *din[1:]

			nxt = create_layer.view()
			nxt.push('din', din, silent=True)
			nxt.push('in_channels', din[0], silent=True)
			nxt.push('out_channels', out_chn, silent=True)
			nxt.push('factor', fct, silent=True)

			layer = next(create_layer)
			layers.append(layer)

			din = layer.dout

		assert din == out_shape, f'error: {din} vs {out_shape}'

		self.branches = nn.ModuleList(branches)
		self.layers = nn.ModuleList(layers)

		assert not split_latent or len(self.branches) == sum(self.branch_dim) == len(self.layers)
		assert split_latent or len(self.branches) == len(self.layers)


	def _visualize(self, out, logger):
		pass


	def forward(self, q):

		if self.root_dim > 0:
			root, q = q[:, :self.root_dim], q[:, self.root_dim:]
			c = self.root(root)
		else:
			B = q.size(0)
			c = self.root
			shape = c.shape
			c = c.expand(B, *shape)

		styles = q.split(self.branch_dim, dim=1) if self.split_latent else [q]*len(self.branches)
		styles = iter(styles)

		for branch, conv in zip(self.branches, self.layers):
			
			if branch is not None:
				style = next(styles)
				c = branch(c, style)
			c = conv(c)

		return c


@fig.Component('ladder')
class Ladder_Branch(lgcy.adain.AdaIN):
	def __init__(self, A):
		
		super().__init__(A)
		
		din = self.din
		qdim = self.style_dim
		ndim = self.noise_dim
		
		extra = qdim if ndim is None else ndim
		if isinstance(din, (tuple, list)):
			dout = extra + din[0], *din[1:]
		else:
			dout = din + extra
		self.dout = dout
	
	def include_noise(self, x, q):
		
		B = q.size(0)
		
		q = q.unsqueeze(-1).unsqueeze(-1).expand(B, self.noise_dim, *x.shape[2:])
		
		return torch.cat([x,q], 1) # concatenates the noise
		
