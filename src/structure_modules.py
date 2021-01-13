import torch
from torch import nn

import omnifig as fig

import foundation as fd
from foundation import models
from foundation import util



def _find_dims(A, din=None, dout=None):

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

@fig.Component('adain')
class AdaIN(models.StyleLayer):
	def __init__(self, A, din=None, dout=None, style_dim=None, **kwargs):

		din, dout = _find_dims(A, din, dout)

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

	def process_style(self, style):
		style = super().process_style(style)
		return self.net(style)

	def infuse(self, content, style):
		if self.norm is not None:
			content = self.norm(content)
		if len(content.shape) != len(style.shape):
			assert len(content.shape) > len(style.shape), 'not the right sizes: {} vs {}'.format(content.shape, style.shape)
			style = style.view(*style.shape, *(1,) * (len(content.shape) - len(style.shape)))
		return content + style


@fig.Component('affine-adain')
class Affine_AdaIN(AdaIN):
	def __init__(self, A, **kwargs):
		net = A.pull('net', None, raw=True, silent=True)
		if net is not None:
			net.push('_mod.normal', 4, silent=True)
		super().__init__(A, **kwargs)

	def infuse(self, content, style):
		if self.norm is not None:
			content = self.norm(content)
		mu, sigma = style.loc, style.scale
		if len(content.shape) != len(mu.shape):
			assert len(content.shape) > len(mu.shape), 'not the right sizes: {} vs {}'.format(content.shape, mu.shape)
			mu = mu.view(*mu.shape, *(1,)*(len(content.shape) - len(mu.shape)))
			sigma = sigma.view(*sigma.shape, *(1,) * (len(content.shape) - len(sigma.shape)))

		return sigma * content + mu


@fig.Component('ladder')
class LadderLayer(models.StyleLayer):
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

	def infuse(self, content, style):
		
		B = style.size(0)
		
		style = style.unsqueeze(-1).unsqueeze(-1).expand(B, self.extra_dim, *content.shape[2:])
		
		return torch.cat([content, style], 1)  # concatenates the noise





