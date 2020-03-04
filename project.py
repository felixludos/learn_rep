
import sys, os, time, shutil#, traceback, ipdb
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import configargparse

import numpy as np
#%matplotlib tk
import matplotlib.pyplot as plt

import umap, shap
import umap.plot
import gpumap
from sklearn.decomposition import PCA

import foundation as fd
from foundation import models
from foundation import util
from foundation import train as trn
from foundation import data

import pointnets
import decoders

MY_PATH = os.path.dirname(os.path.abspath(__file__))

trn.register_config_dir(os.path.join(MY_PATH, 'config'), recursive=True)

# region Algorithms

@fd.Component('ae')
class AutoEncoder(fd.Generative, fd.Encodable, fd.Decodable, fd.Regularizable, fd.Schedulable,
                  fd.Cacheable, fd.Visualizable, fd.Trainable_Model):
	def __init__(self, A):

		encoder = A.pull('encoder')
		decoder = A.pull('decoder')

		criterion = A.pull('criterion', 'bce') # {'_type':'criterion', 'name':'bce', 'kwargs':{'reduction':'sum'}}

		reg_wt = A.pull('reg_wt', 0)
		reg = A.pull('reg', 'L2')

		viz_gen = A.pull('viz_gen', False)

		super().__init__(encoder.din, decoder.dout)

		self.enc = encoder
		self.dec = decoder

		self.criterion = util.get_loss_type(criterion, reduction='sum')
		self.reg_wt = reg_wt
		self.reg_fn = get_regularization(reg, reduction='sum')
		if self.reg_wt > 0:
			self.stats.new('reg')

		self.set_optim()
		self.set_scheduler()

		self.register_buffer('_q', None)
		self.register_cache('_real', None)
		self.register_cache('_rec', None)
		self.viz_gen = viz_gen

	def _visualize(self, info, logger):

		if isinstance(self.enc, fd.Visualizable):
			self.enc.visualize(info, logger)
		if isinstance(self.dec, fd.Visualizable):
			self.dec.visualize(info, logger)

		if self._viz_counter % 2 == 0:
			if 'latent' in info and info.latent is not None:
				q = info.latent.loc if isinstance(info.latent, distrib.Distribution) else info.latent

				shape = q.size()
				if len(shape) > 1 and np.product(shape) > 0:
					try:
						logger.add('histogram', 'latent-norm', q.norm(p=2, dim=-1))
						logger.add('histogram', 'latent-std', q.std(dim=0))
					except ValueError:
						print('\n\n\nWARNING: histogram just failed\n')
						print(q.shape, q.norm(p=2, dim=-1).shape)

			B, C, H, W = info.original.shape
			N = min(B, 8)

			if 'reconstruction' in info:
				viz_x, viz_rec = info.original[:N], info.reconstruction[:N]

				recs = torch.cat([viz_x, viz_rec], 0)
				logger.add('images', 'rec', self._img_size_limiter(recs))
			elif self._rec is not None:
				viz_x, viz_rec = self._real[:N], self._rec[:N]

				recs = torch.cat([viz_x, viz_rec], 0)
				logger.add('images', 'rec', self._img_size_limiter(recs))

			if self.viz_gen:
				viz_gen = self.generate(2*N)
				logger.add('images', 'gen', self._img_size_limiter(viz_gen))

			logger.flush()

	def _img_size_limiter(self, imgs):
		H, W = imgs.shape[-2:]

		if H*W < 2e4: # allows upto around 128x128
			return imgs

		imgs = F.interpolate(imgs, size=(128,128))
		return imgs

	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		x = batch[0]
		B = x.size(0)

		out.original = x

		rec, q = self(x, ret_q=True)
		out.latent = q
		out.reconstruction = rec

		self._rec, self._real = rec.detach(), x.detach()

		loss = self.criterion(rec, x) / B
		out.rec_loss = loss

		if self.reg_wt > 0:
			reg_loss = self.regularize(q)
			self.stats.update('reg', reg_loss)
			out.reg_loss = reg_loss
			loss += self.reg_wt * reg_loss

		out.loss = loss

		if self.train_me():
			self._q = q.loc.detach() if isinstance(q, distrib.Normal) else q.detach()

			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

		return out


	def hybridize(self, q=None):

		if q is None:
			q = self._q

		return util.shuffle_dim(q)

	def generate(self, N=1):

		if self._q is None:
			raise NotImplementedError

		q = torch.cat([self._q]*(N//len(self._q)+1))

		hyb = self.hybridize(q)[:N]

		return self.decode(hyb)

	def encode(self, x):
		return self.enc(x)

	def decode(self, q):
		return self.dec(q)

	def forward(self, x, ret_q=False):

		q = self.encode(x)
		rec = self.decode(q)

		if ret_q:
			return rec, q
		return rec

	def regularize(self, q):
		B = q.size(0)
		mag = self.reg_fn(q)
		return mag / B

@fd.Component('vae')
class VAE(AutoEncoder):
	def __init__(self, A):

		assert 'reg_wt' in A and A.reg_wt > 0, 'not a vae without regularization'

		A.reg = None

		super().__init__(A)

		assert isinstance(self.enc, models.Normal), 'encoder must output a normal distrib'

	def regularize(self, q):
		return util.standard_kl(q).sum().div(q.loc.size(0))

	def decode(self, q):
		if isinstance(q, distrib.Distribution):
			q = q.rsample()
		return super().decode(q)


@fd.AutoComponent('regularization')
def get_regularization(name, p=2, dim=1, reduction='mean', **kwargs):

	if not isinstance(name, str):
		return name

	if name == 'L2' or name =='l2':
		return util.Lp_Norm(p=2, dim=dim, reduction=reduction)
	elif name == 'L1' or name == 'l1':
		return util.Lp_Norm(p=1, dim=dim, reduction=reduction)
	elif name == 'Lp':
		return util.Lp_Norm(p=p, dim=dim, reduction=reduction)
	else:
		print(f'Unknown reg: {name}')
		# raise Exception(f'unknown: {name}')





# endregion


# region Architectures


@fd.Component('extraction-enc')
class UMAP_Encoder(fd.Encodable, fd.Schedulable, fd.Model):

	def __init__(self, A):

		in_shape = A.pull('in_shape', '<>din')
		latent_dim = A.pull('latent_dim', '<>dout')
		feature_dim = A.pull('feature_dim', '<>latent_dim')

		transform = A.pull('transform', None)

		alg = A.pull('alg', 'umap')

		kwargs = {
			'n_components': feature_dim,
		}

		if alg == 'umap':

			extraction_cls = gpumap.GPUMAP

			kwargs['random_state'] = A.pull('random_state', '<>seed')
			kwargs['min_dist'] = A.pull('min_dist', 0.1)
			kwargs['n_neighbors'] = A.pull('neighbors', 15)

		elif alg == 'pca':
			extraction_cls = PCA

		else:
			raise Exception(f'unknown alg: {alg}')

		extractor = extraction_cls(**kwargs)

		if 'net' in A:
			A.net.din = feature_dim
			A.net.dout = latent_dim

		net = A.pull('net', None)

		training_limit = A.pull('training_limit', None)

		super().__init__(din=in_shape, dout=feature_dim if net is None else latent_dim)

		self.training_limit = training_limit

		self.transformer = transform

		self.alg = alg
		self.extractor = extractor

		self.net = net

		self.set_optim(A)
		self.set_scheduler(A)

	def _resize(self, x):
		N, C, H, W = x.shapes

		if H >= 64:
			return x[:, :, ::2, ::2].reshape(N, -1)
		return x.reshape(N, -1)

	def prep(self, traindata, *other):

		samples = traindata.get_raw_data().float()

		if self.training_limit is not None:
			samples = samples[:self.training_limit]

		samples = self._reformat(samples)

		print(f'Training a {self.alg} feature extractor to extract {self.extractor.n_components} '
		      f'features from an input {samples.shape}')


		# fit estimator
		self.extractor.fit(samples)

		print('Feature extraction complete')

	def encode(self, x):
		return self(x)

	def transform(self, x):

		device = x.device
		x = self._reformat(x)

		q = self.extractor.transform(x)
		q = torch.from_numpy(q).to(device)

		return q

	def _reformat(self, x):
		x = x.cpu().numpy()

		if self.transformer is not None:
			x = self.transformer(x)
		else:
			x = self._resize(x)

		return x

	def forward(self, x):

		q = self.transform(x)

		if self.net is None:
			return q
		return self.net(q)






@fd.Component('dislib-enc')
class Disentanglement_lib_Encoder(fd.Encodable, fd.Schedulable, fd.Model):
	def __init__(self, A):

		in_shape = A.pull('in_shape', '<>din')
		latent_dim = A.pull('latent_dim', '<>dout')

		nonlin = A.pull('nonlin', 'relu')

		C, H, W = in_shape

		assert (H,W) in {(64,64), (128,128)}, f'not a valid input size: {(H,W)}'

		net_type = A.pull('net_type', 'conv')

		assert net_type in {'conv', 'fc'}, f'unknown type: {net_type}'

		super().__init__(din=in_shape, dout=latent_dim)

		if net_type == 'conv':

			channels = [32,32,32,64,64]
			kernels = [4,4,4,2,2]
			strides = [2,2,2,2,2]

			if H == 64:
				channels = channels[1:]
				kernels = kernels[1:]
				strides = strides[1:]

			shapes, settings = models.plan_conv(in_shape, channels=channels, kernels=kernels, strides=strides)

			out_shape = shapes[-1]

			self.conv = nn.Sequential(*models.build_conv_layers(settings, nonlin=nonlin, out_nonlin=nonlin,
			                                                   pool_type=None, norm_type=None))

			self.net = models.make_MLP(out_shape, latent_dim, hidden_dims=[256,], nonlin=nonlin)

		else:

			self.net = models.make_MLP(in_shape, latent_dim, hidden_dims=[1200, 1200], nonlin=nonlin)

		self.uses_conv = net_type == 'conv'

		self.set_optim(A)
		self.set_scheduler(A)

	def forward(self, x):
		c = self.conv(x) if self.uses_conv else x
		q = self.net(c)
		return q

	def encode(self, x):
		return self(x)


@fd.Component('dislib-dec')
class Disentanglement_lib_Decoder(fd.Decodable, fd.Schedulable, fd.Model):
	def __init__(self, A):

		latent_dim = A.pull('latent_dim', '<>din')
		out_shape = A.pull('out_shape', '<>dout')

		nonlin = A.pull('nonlin', 'relu')

		C, H, W = out_shape

		assert (H, W) in {(64, 64), (128, 128)}, f'not a valid output size: {(H, W)}'

		net_type = A.pull('net_type', 'conv')

		assert net_type in {'conv', 'fc'}, f'unknown type: {net_type}'

		super().__init__(din=latent_dim, dout=out_shape)

		if net_type == 'conv':

			channels = [64, 64, 32, 32, 32]
			kernels = [4, 4, 4, 4, 4]
			strides = [2, 2, 2, 2, 2]

			if H == 64:
				channels = channels[:-1]
				kernels = kernels[:-1]
				strides = strides[:-1]

			shapes, settings = models.plan_deconv(out_shape, channels=channels, kernels=kernels, strides=strides)

			in_shape = shapes[0]

			self.net = models.make_MLP(latent_dim, in_shape, hidden_dims=[256], nonlin=nonlin, )

			self.deconv = nn.Sequential(*models.build_deconv_layers(settings, sizes=shapes[:-1],
			                                                        nonlin=nonlin, out_nonlin='sigmoid',
			                                                        norm_type=None))

		else:

			self.net = models.make_MLP(latent_dim, out_shape, hidden_dims=[1200,1200,1200], nonlin=nonlin)

		self.uses_conv = net_type == 'conv'

		self.set_optim(A)
		self.set_scheduler(A)

	def forward(self, q):
		c = self.net(q)
		x = self.deconv(c) if self.uses_conv else c
		return x

	def decode(self, q):
		return self(q)

# endregion


# region Sup-Models

@fd.Component('sup-model')
class SupModel(fd.Visualizable, fd.Trainable_Model):
	def __init__(self, A):

		net = A.pull('net')
		criterion = A.pull('criterion', 'cross-entropy')

		super().__init__(net.din, net.dout)

		self.net = net
		self.criterion = util.get_loss_type(criterion)

		self.stats.new('error', 'confidence')

		self.set_optim(A)


	def forward(self, x):
		return self.net(x)

	def _visualize(self, info, logger):
		# if self._viz_counter % 5 == 0:
		# 	pass

		conf, pick = info.pred.max(-1)

		confidence = conf.detach()
		correct = pick.sub(info.y).eq(0).float().detach()

		self.stats.update('confidence', confidence.mean())
		self.stats.update('error', 1 - correct.mean())


	def _step(self, batch, out=None):
		if out is None:
			out = util.TensorDict()

		# compute loss
		x, y = batch

		out.x, out.y = x, y

		pred = self(x)
		out.pred = pred

		loss = self.criterion(pred, y)
		out.loss = loss

		if self.train_me():
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

		return out

# endregion

if __name__ == '__main__':
	sys.exit(trn.main(argv=sys.argv))



