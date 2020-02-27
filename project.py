
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

import foundation as fd
from foundation import models
from foundation import util
from foundation import train as trn
from foundation import data

MY_PATH = os.path.dirname(os.path.abspath(__file__))

trn.register_config_dir(os.path.join(MY_PATH, 'config'), recursive=True)


@fd.Component('ae')
class AutoEncoder(fd.Generative, fd.Encodable, fd.Decodable, fd.Regularizable, fd.Schedulable,
                  fd.Cacheable, fd.Visualizable, fd.Trainable_Model):
	def __init__(self, A):

		encoder = A.pull('encoder')
		decoder = A.pull('decoder')

		criterion = A.pull('criterion', 'bce') # {'_type':'criterion', 'name':'bce', 'kwargs':{'reduction':'sum'}}

		reg_wt = A.pull('reg_wt', 0)

		viz_gen = A.pull('viz_gen', False)

		super().__init__(encoder.din, decoder.dout)

		self.enc = encoder
		self.dec = decoder

		self.criterion = util.get_loss_type(criterion, reduction='sum')
		self.reg_wt = reg_wt

		self.stats.new('reg')

		self.set_optim()
		self.set_scheduler()

		self.register_buffer('_q', None)
		self.register_cache('_real', None)
		self.register_cache('_rec', None)
		self.viz_gen = viz_gen

	def _visualize(self, info, logger):

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

		out.x = x

		rec, q = self(x, ret_q=True)
		out.latent = q
		out.reconstruction = rec

		self._rec, self._real = rec.detach(), x.detach()

		loss = self.criterion(rec, x) / B
		out.rec_loss = loss

		if self.reg_wt > 0:
			reg_loss = self.regularize(q)
			out.reg_loss = reg_loss
			loss += self.reg_wt * reg_loss

		out.loss = loss

		if self.train_me():
			self._q = q.detach()

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

	def regularize(self, q, p=None): # simple L2 regularization
		return q.pow(2).sum().div(q.size(0))



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

if __name__ == '__main__':
	sys.exit(trn.main(argv=sys.argv))



