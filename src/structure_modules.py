import torch
from torch import nn

import omnifig as fig

import foundation as fd
from foundation import models
from foundation import util

class StyleLayer(fd.Function):

	def __init__(self, A, style_dim=None, **kwargs):

		if style_dim is None:
			style_dim = A.pull('style-dim', None)

		super().__init__(A, **kwargs)

		self.style_dim = style_dim
		self.register_buffer('_style', None)

	def _find_dims(self, A, din=None, dout=None):

		if din is None and dout is None:
			din = A.pull('din', None)
			if din is None:
				dout = A.pull('dout')
				din = dout
			else:
				dout = din
		elif din is not None:
			dout = din
		elif dout is not None:
			din = dout
		else:
			assert din == dout, f'bad size: {din} vs {dout}'

		return din, dout

	def cache_style(self, q):
		self._style = q

	def _process_style(self, q):

		if q is None:
			assert self._style is not None, 'no style provided or cached'
			q = self._style
		elif self._style is not None:
			raise Exception('Provided a style, when one was already cached')

		self._style = None

		return q

	def infuse(self, x, q):
		raise NotImplementedError

	def forward(self, x, q=None):

		q = self._process_style(q)

		return self.infuse(x, q)

class PriorStructuralLayer(models.Prior, StyleLayer):

	def __init__(self, A, style_dim=None, **kwargs):
		super().__init__(A, style_dim=style_dim, prior_dim=style_dim, **kwargs)
		if style_dim is None:
			self.prior_dim = self.style_dim

	def forward(self, x, q=None):
		if q is None and self._style is None:
			self.cache_style(self.sample_prior(x.size(0)))
		return super().forward(x, q=q)

@fig.AutoModifier('gaussian-style')
class Gaussian(models.Gaussian, PriorStructuralLayer):
	pass

@fig.AutoModifier('uniform-style')
class Uniform(models.Uniform, PriorStructuralLayer):
	pass


@fig.Component('adain')
class AdaIN(StyleLayer):
	def __init__(self, A, din=None, dout=None, style_dim=None, **kwargs):

		din, dout = self._find_dims(A, din, dout)

		if style_dim is None:
			style_dim = A.pull('style-dim', None)

		feature_dim = din[0] if isinstance(din, (list, tuple)) and not isinstance(style_dim, (list, tuple)) else din
		
		net_info = A.pull('net', None, silent=True, raw=True)
		if net_info is not None:
			net_info.push('din', style_dim, silent=True)
			net_info.push('latent-dim', style_dim, silent=True)
			net_info.push('dout', feature_dim, silent=True)

		net = A.pull('net', None)

		if net is None and style_dim != feature_dim:
			net = models.make_MLP(style_dim, feature_dim)

		use_norm = A.pull('normalize-features', True)
		norm = util.get_normalization('instance', feature_dim, affine=False) if use_norm else None

		super().__init__(A, din=din, style_dim=style_dim, dout=dout, **kwargs)

		self.norm = norm
		self.net = net
		self.feature_dim = feature_dim

	def _process_style(self, q):
		q = super()._process_style(q)
		return self.net(q)

	def infuse(self, x, q):
		if self.norm is not None:
			x = self.norm(x)
		if len(x.shape) != len(q.shape):
			assert len(x.shape) > len(q.shape), 'not the right sizes: {} vs {}'.format(x.shape, q.shape)
			q = q.view(*q.shape, *(1,)*(len(x.shape)-len(q.shape)))
		return x + q


@fig.Component('norm-adain')
class Norm_AdaIN(AdaIN):
	def __init__(self, A, **kwargs):
		net = A.pull('net', None, raw=True, silent=True)
		if net is not None:
			net.push('_mod.normal', 4, silent=True)
		super().__init__(A, **kwargs)

	def include_noise(self, x, q):
		if self.norm is not None:
			x = self.norm(x)
		mu, sigma = q.loc, q.scale
		if len(x.shape) != len(mu.shape):
			assert len(x.shape) > len(mu.shape), 'not the right sizes: {} vs {}'.format(x.shape, mu.shape)
			mu = mu.view(*mu.shape, *(1,)*(len(x.shape)-len(mu.shape)))
			sigma = sigma.view(*sigma.shape, *(1,) * (len(x.shape) - len(sigma.shape)))

		return sigma*x + mu


@fig.Component('ladder')
class LadderLayer(StyleLayer):
	def __init__(self, A, din=None, dout=None, style_dim=None, **kwargs):

		if style_dim is None:
			style_dim = A.pull('style-dim')

		din, dout = self._find_dims(A, din, dout)
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

	def _process_style(self, q):
		q = super()._process_style(q)
		if self.net is not None:
			q = self.net(q)
		return q

	def include_noise(self, x, q):
		
		B = q.size(0)
		
		q = q.unsqueeze(-1).unsqueeze(-1).expand(B, self.extra_dim, *x.shape[2:])
		
		return torch.cat([x, q], 1)  # concatenates the noise





