
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

try:
	import umap, shap
	import umap.plot
	import gpumap
except ImportError:
	print('WARNING: umap not found')
from sklearn.decomposition import PCA

import foundation as fd
from foundation import models
from foundation import util
from foundation import train as trn
from foundation import data

import visualizations as viz_util

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

		hparams = {'reg_type': str(reg),}

		super().__init__(encoder.din, decoder.dout)

		self.enc = encoder
		self.dec = decoder
		
		self.latent_dim = self.enc.dout

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
		
		self._hparams = hparams

	def get_hparams(self):
		
		h = self.enc.get_hparams()
		h.update(self.dec.get_hparams())
		
		h['reg_wt'] = self.reg_wt
		
		h.update(self._hparams)
		
		return h

	def _visualize(self, info, logger):

		if isinstance(self.enc, fd.Visualizable):
			self.enc.visualize(info, logger)
		if isinstance(self.dec, fd.Visualizable):
			self.dec.visualize(info, logger)

		if self._viz_counter % 2 == 0 or not self.training:
			q = None
			if 'latent' in info and info.latent is not None:
				q = info.latent.loc if isinstance(info.latent, distrib.Distribution) else info.latent

				shape = q.size()
				if len(shape) > 1 and np.product(shape) > 0:
					try:
						logger.add('histogram', 'latent-norm', q.norm(p=2, dim=-1))
						logger.add('histogram', 'latent-std', q.std(dim=0))
						if isinstance(info.latent, distrib.Distribution):
							logger.add('histogram', 'std-hist', info.latent.scale.mean(dim=0))
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

			if not self.training: # expensive visualizations
				
				n = 16
				steps = 20
				ntrav = 1
				
				if q is not None and len(q) >= n:
					fig, (lax, iax) = plt.subplots(2, figsize=(3*min(q.size(1)//20+1,3),3))
					
					viz_util.viz_latent(q, figax=(fig, lax), )
					
					Q = q[:n]
					
					vecs = viz_util.get_traversal_vecs(Q, steps=steps,
                          mnmx=(Q.min(0)[0].unsqueeze(-1), Q.max(0)[0].unsqueeze(-1))).contiguous()
					# deltas = torch.diagonal(vecs, dim1=-3, dim2=-1)
					
					walks = viz_util.get_traversals(vecs, self.decode).cpu()
					diffs = viz_util.compute_diffs(walks)
					
					viz_util.viz_interventions(diffs, figax=(fig, iax))
					

					# fig.tight_layout()
					border, between = 0.02, 0.01
					plt.subplots_adjust(wspace=between, hspace=between,
											left=3*border, right=1 - border, bottom=border, top=1 - border)
					
					logger.add('figure', 'distrib', fig)
					
					full = walks[:ntrav]
					del walks
					
					tH, tW = util.calc_tiling(full.size(1))
					B, N, S, C, H, W = full.shape
					
					# if tH*H > 200: # limit total image size
					# 	pass
					
					full = full.view(B, tH, tW, S, C, H, W)
					full = full.permute(0, 3, 4, 1, 5, 2, 6).contiguous().view(B, S, C, tH * H, tW * W)
					
					logger.add('video', 'traversals', full, fps=12)
				
				
				else:
					print('WARNING: visualizing traversals failed')
					
				
				pass

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

class Prior_Autoencoder(AutoEncoder):
	
	def sample_prior(self, N=1):
		return torch.randn(N, self.latent_dim, device=self.device)
	
	def generate(self, N=1):
		q = self.sample_prior(N)
		return self.decode(q)

@fd.Component('vae')
class VAE(Prior_Autoencoder):
	def __init__(self, A, norm_mod=None):
		
		if norm_mod is None:
			norm_mod = models.Normal

		assert 'reg_wt' in A and A.reg_wt > 0, 'not a vae without regularization'

		A.reg = None

		super().__init__(A)
		
		self._hparams['reg_type'] = 'KL'
		self._hparams['enc_type'] = 'VAE'

		if not isinstance(self.enc, norm_mod):
			print('WARNING: encoder apparently does not output a normal distribution')
		# assert isinstance(self.enc, models.Normal), 'encoder must output a normal distrib'

	def regularize(self, q):
		return util.standard_kl(q).sum().div(q.loc.size(0))

	def decode(self, q):
		if isinstance(q, distrib.Distribution):
			q = q.rsample()
		return super().decode(q)

@fd.Component('wae')
class WAE(Prior_Autoencoder):
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._hparams['reg_type'] = 'W'
	
	def regularize(self, q, p=None):
		if p is None:
			p = self.sample_prior(q.size(0))
		return util.MMD(p, q)

@fd.Component('swae')
class Slice_WAE(Prior_Autoencoder):
	def __init__(self, A):
		slices = A.pull('slices', '<>latent_dim')

		super().__init__(A)

		self.slices = slices
		
		self._hparams['reg_type'] = 'SW'
		self._hparams['slices'] = slices

	def sample_slices(self, N=None): # sampled D dim unit vectors
		if N is None:
			N = self.slices

		return F.normalize(torch.randn(self.latent_dim, N, device=self.device), p=2, dim=0)

	def regularize(self, q, p=None):

		s = self.sample_slices() # D, S

		qd = q @ s
		qd = qd.sort(0)[0]

		if p is None:
			p = self.sample_prior(q.size(0))
		pd = p @ s
		pd = pd.sort(0)[0]

		return (qd - pd).abs().mean()


class Cost_Aware(Prior_Autoencoder):
	def __init__(self, A):
		
		reg_imp_p = A.pull('reg_imp_p', 1)
		reg_imp_wt = A.pull('reg_imp_wt', 0.5)
		
		init_imp_mu = A.pull('init_imp_mu', 0)
		init_imp_std = A.pull('init_imp_std', 1)
		
		weigh_distances = A.pull('weigh_distances', False)
		
		super().__init__(A)
		
		self.stats.new('reg_imp', 'reg_prior')
		
		self.importance = nn.Parameter(init_imp_std*torch.randn(self.latent_dim) + init_imp_mu,
		                               requires_grad=True)

		self.reg_imp_p = reg_imp_p
		self.reg_imp_wt = reg_imp_wt
		
		self.weigh_distances = weigh_distances

	def get_importance(self):
		return F.sigmoid(self.importance)

	def regularize(self, q):
		
		v = self.get_importance()
		
		reg_imp = v.norm(p=self.reg_imp_p)
		self.stats.update('reg_imp', reg_imp)
		
		p = self.sample_prior(q.size(0))
		if self.weigh_distances:
			q = q * v.unsqueeze(0)
			p = p * v.unsqueeze(0)
		
		reg_prior = super().regularize(q, p)
		self.stats.update('reg_prior', reg_prior)
		
		return self.reg_imp_wt * reg_imp + (1 - self.reg_imp_wt) * reg_prior


@fd.Component('cae')
class Det_Cost_Aware(Cost_Aware):
	def encode(self, x):
		q = super().encode(x)
		B, D = q.size()
		
		v = self.get_importance().expand(B, D)
		p = self.sample_prior(B)
		
		q = v * q + (1 - v) * p
		return q

class Sto_Cost_Aware(Cost_Aware):
	def encode(self, x):
		q = super().encode(x)
		return self.as_normal(q)
	
	def as_normal(self, q):
		std = self.get_importance().expand(*q.size())
		return distrib.Normal(loc=q, scale=std)

@fd.Component('cwae')
class Cost_Aware_WAE(Det_Cost_Aware, WAE):
	pass

@fd.Component('scwae')
class Cost_Aware_SWAE(Det_Cost_Aware, Slice_WAE):
	pass

@fd.Component('cvae')
class Cost_VAE(Sto_Cost_Aware, VAE):
	pass


@fd.AutoModifier('fixed-std')
class Fixed_Std(fd.Visualizable, fd.Model):
	def __init__(self, A, latent_dim=None):
		
		if latent_dim is None:
			latent_dim = A.pull('latent_dim', '<>dout')
		
		min_log_std = A.pull('min_log_std', None)
		
		super().__init__(A)
		
		self.log_std = nn.Parameter(torch.randn(latent_dim), requires_grad=True)
		
		self.min_log_std = min_log_std
		self.latent_dim = latent_dim
		
	def _visualize(self, info, logger):

		try:
			super()._visualize(info, logger)
		except NotImplementedError:
			pass


		pass
	
	def forward(self, *args, **kwargs):
		
		mu = super().forward(*args, **kwargs)
		logsigma = self.log_std
		
		if self.min_log_std is not None:
			logsigma = logsigma.clamp(min=self.min_log_std)

		return distrib.Normal(loc=mu, scale=logsigma.exp())
		

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



