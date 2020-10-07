import torch
from torch import nn

import omnifig as fig

# from ..old import train as trn
import foundation as fd
from foundation import models


@fig.Component('adain')
class AdaIN(fd.Model):
	def __init__(self, A):

		qdim = A.pull('style_dim', '<>din') # note: terminologywise style_dim and noise_dim should be swapped
		cdim = A.pull('features', '<>dout')

		pixelwise = A.pull('pixelwise', False)

		ndim = A.pull('noise_dim', cdim[0] if isinstance(cdim, (tuple, list)) and not pixelwise else cdim)

		ntype = A.pull('net._type', None, silent=True)
		if ntype is not None:
			net_config = A['net']
			net_config.push('din', qdim)
			net_config.push('dout', ndim)
			net_config.push('latent_dim', ndim)

		net = A.pull('net', None)

		if net is None and qdim != ndim:
			net = nn.Linear(qdim, ndim)

		if isinstance(qdim, (tuple,list)) and isinstance(cdim, (tuple,list)):
			raise NotImplementedError

		super().__init__(cdim, cdim)

		self.net = net
		self.style_dim = qdim
		self.noise_dim = ndim
		self.noise = None

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

@fig.Component('norm-ada-in')
class Norm_AdaIN(AdaIN):
	def __init__(self, A):
		net = A['net']
		ntype = net.pull('_type', None, silent=True)
		if ntype is not None:
			net.push('_mod', 'normal', silent=True)
		super().__init__(A)

	def include_noise(self, x, q):
		mu, sigma = q.loc, q.scale
		if len(x.shape) != len(mu.shape):
			assert len(x.shape) > len(mu.shape), 'not the right sizes: {} vs {}'.format(x.shape, mu.shape)
			mu = mu.view(*mu.shape, *(1,)*(len(x.shape)-len(mu.shape)))
			sigma = sigma.view(*sigma.shape, *(1,) * (len(x.shape) - len(sigma.shape)))

		return sigma*x + mu




@fig.Component('adain-double-dec')
class AdaIn_Double_Decoder(models.Double_Decoder):

	def __init__(self, A):

		adain_latent_dim = A.pull('adain_latent_dim', 0)
		full_latent_dim = A.pull('latent_dim', '<>din')

		const_start = False
		init_latent_dim = full_latent_dim - adain_latent_dim
		assert init_latent_dim >= 0, f'invalid: {full_latent_dim}'
		if init_latent_dim == 0:
			init_latent_dim = 1
			const_start = True
		A.push('latent_dim', init_latent_dim)
		A.push('full_latent_dim', full_latent_dim)
		A.push('adain_latent_dim', adain_latent_dim)

		super().__init__(A)

		A.push('latent_dim', full_latent_dim)

		if adain_latent_dim is not None:
			self.din = full_latent_dim

		self.adain_latent_dim = adain_latent_dim

		self.init_latent_dim = init_latent_dim
		self.const_start = const_start

	def _create_layers(self, chns, factors, internal_channels, squeeze, A):

		between_blocks = len(chns)-2
		adains = A.pull('adains', [True] * between_blocks)
		try:
			len(adains)
		except TypeError:
			adains = [adains]
		if len(adains) != between_blocks:
			adains = adains * between_blocks
		adains = iter(adains)

		splits = A.pull('splits', None)
		if splits is not None:
			try:
				len(splits)
			except TypeError:
				splits = [splits]
			if len(splits) != between_blocks:
				splits = splits * between_blocks
			splits = splits[:between_blocks]

		self.splits = splits
		if splits is not None:
			splits = iter(splits)
		full_latent = A.pull('full_latent_dim')

		nonlin = A.pull('nonlin', 'elu')
		output_nonlin = A.pull('output_nonlin', None)
		output_norm_type = A.pull('output_norm_type', None)

		up_type = A.pull('up_type', 'bilinear')
		norm_type = A.pull('norm_type', None)
		residual = A.pull('residual', False)

		last_chn = chns[-2:]
		chns = chns[:-1]

		layers = []

		alayers = []

		for ichn, ochn in zip(chns, chns[1:]):
			layers.append(
				models.DoubleDeconvLayer(in_channels=ichn, out_channels=ochn, factor=next(factors),
				                            up_type=up_type, norm_type=norm_type,
				                            nonlin=nonlin, output_nonlin=nonlin,
				                            internal_channels=next(internal_channels), squeeze=next(squeeze),
				                            residual=residual,
				                            )
			)
			if next(adains):

				dim = next(splits) if splits is not None else full_latent
				if dim is not None:
					A.adain.ada_noise = dim
					A.adain.features = ochn

				adain = A.pull('adain', force_new=True)

				if 'ada_noise' in A.adain:
					del A.adain.ada_noise
				if 'features' in A.adain:
					del A.adain.features

				alayers.append(adain)
				layers.append(adain)
		layers.append(
			models.DoubleDeconvLayer(in_channels=last_chn[0], out_channels=last_chn[1], factor=next(factors),
			                            up_type=up_type, norm_type=output_norm_type,
			                            nonlin=nonlin, output_nonlin=output_nonlin,
			                            internal_channels=next(internal_channels), squeeze=next(squeeze),
			                            residual=residual,
			                            )
		)

		self.ada_ins = alayers

		return nn.ModuleList(layers)

	def forward(self, q):

		if self.const_start:
			init = torch.ones(q.size(0), 1, dtype=q.dtype, device=q.device)
		else:
			init, q = q[...,:self.init_latent_dim], q[...,self.init_latent_dim:]

		noises = [q]*len(self.ada_ins)
		if self.splits is not None:
			noises = torch.split(q,self.splits, dim=-1)

		for adain, noise in zip(self.ada_ins, noises):
			adain.set_noise(noise)

		return super().forward(init)
