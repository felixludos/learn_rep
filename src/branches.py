import torch
from torch import nn

import omnifig as fig

import foundation as fd
from foundation import models


@fig.Component('adain')
class AdaIN(fd.Function):
	def __init__(self, A, din=None, dout=None, style_dim=None, **kwargs):
		
		if din is None and dout is None:
			din = A.pull('din', None)
			if din is None:
				dout = A.pull('dout')
		elif din is not None:
			dout = din
		elif dout is not None:
			din = dout
		else:
			assert din == dout, f'bad size: {din} vs {dout}'

		feature_dim = din[0] if isinstance(din, (list, tuple)) and not isinstance(style_dim, (list, tuple)) else din

		if style_dim is None:
			style_dim = A.pull('style_dim', None)
		
		net_info = A.pull('net', None, silent=True, raw=True)
		if net_info is not None:
			net_info.push('din', style_dim)
			net_info.push('latent_dim', style_dim)
			net_info.push('dout', feature_dim)

		net = A.pull('net', None)

		if net is None and style_dim != feature_dim:
			net = models.make_MLP(style_dim, feature_dim)

		super().__init__(A, din=din, dout=dout, **kwargs)

		self.net = net
		self.style_dim = style_dim
		self.feature_dim = feature_dim
		self.register_buffer('noise', None)

	def default_noise(self, n):
		return torch.zeros(n.size(0), *self.dout, device=n.device, dtype=n.dtype)

	def process_noise(self, n):
		if self.net is None:
			return self.default_noise(n)
		return self.net(n)

	def include_noise(self, x, q):
		if len(x.shape) != len(q.shape):
			assert len(x.shape) > len(q.shape), 'not the right sizes: {} vs {}'.format(x.shape, q.shape)
			q = q.view(*q.shape, *(1,)*(len(x.shape)-len(q.shape)))
		return x + q

	def set_noise(self, n):
		self.noise = n

	def forward(self, x, n=None):
		if n is None and self.noise is not None:
			n = self.noise
			self.noise = None
		if n is not None:
			q = self.process_noise(n)
			x = self.include_noise(x, q)
		return x


@fig.Component('norm-adain')
class Norm_AdaIN(AdaIN):
	def __init__(self, A, **kwargs):
		net = A.pull('net', None, raw=True, silent=True)
		if net is not None:
			net.push('_mod.normal', 4, silent=True)
		super().__init__(A, **kwargs)

	def include_noise(self, x, q):
		mu, sigma = q.loc, q.scale
		if len(x.shape) != len(mu.shape):
			assert len(x.shape) > len(mu.shape), 'not the right sizes: {} vs {}'.format(x.shape, mu.shape)
			mu = mu.view(*mu.shape, *(1,)*(len(x.shape)-len(mu.shape)))
			sigma = sigma.view(*sigma.shape, *(1,) * (len(x.shape) - len(sigma.shape)))

		return sigma*x + mu


@fig.Component('ladder')
class Ladder_Branch(AdaIN):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		
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
		
		return torch.cat([x, q], 1)  # concatenates the noise





