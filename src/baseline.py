import sys, os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch import distributions as distrib
from torch.nn import functional as F

from omnibelt import unspecified_argument

import omnifig as fig

import omnilearn as learn
from omnilearn import models
from omnilearn import util

from .methods import VAE



@fig.Component('dislib-enc-arch')
class Disentanglement_lib_Encoder_Arch(learn.Encodable, learn.Optimizable, learn.Function):
	def __init__(self, A, din=None, dout=None, hidden_dim=None, ret_normal=None, **kwargs):
		
		if din is None:
			din = A.pull('in-shape', '<>din')
		if dout is None:
			dout = A.pull('latent-dim', '<>dout')
		
		if ret_normal is None:
			ret_normal = A.pull('ret-normal', True)
		
		latent_dim = dout // 2
		
		C, H, W = din
		
		flat_shape = 64, H // 2 ** 4, W // 2 ** 4
		flat_dim = int(np.product(flat_shape))
		
		if hidden_dim is None:
			hidden_dim = A.pull('hidden_dim', 256)
		
		conv = nn.Sequential(
			nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity(A.pull('nonlin', 'relu')),
			nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.Conv2d(32, 64, kernel_size=2, stride=2),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.Conv2d(64, 64, kernel_size=2, stride=2),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			# nn.Flatten(),
		)
		fc = nn.Sequential(
			nn.Linear(flat_dim, hidden_dim),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
		)
		
		super().__init__(A, din=din, dout=dout, **kwargs)
		
		self.conv = conv
		self.fc = fc
		
		self.mu_layer = nn.Linear(hidden_dim, latent_dim)
		self.logsigma_layer = nn.Linear(hidden_dim, latent_dim)
		
		self.ret_normal = ret_normal and not isinstance(self, models.Normal)
	
	def forward(self, x):
		c = self.conv(x)
		c = c.permute(0,2,3,1).reshape(c.size(0), -1)
		c = self.fc(c)
		
		mu = self.mu_layer(c)
		logsigma = self.logsigma_layer(c)
		
		if self.ret_normal:
			return distrib.Normal(mu, logsigma.exp())
		return torch.cat([mu, logsigma], -1)
		
	
	def encode(self, x):
		return self(x)
	
@fig.Component('dislib-enc')
class Disentanglement_lib_Encoder(learn.TensorflowPort, Disentanglement_lib_Encoder_Arch):
	def __init__(self, A, tf_path=unspecified_argument, din=None, dout=None, **kwargs):
		
		if tf_path is unspecified_argument:
			tf_path = self._process_tf_path(A, tf_path=tf_path)
		if tf_path is not None:
			din, dout = (3, 64, 64), 20
	
		super().__init__(A, tf_path=tf_path, din=din, dout=dout, **kwargs)
	
	# def _convert_tf_val(self, my_name, param, tf_val):
	# 	if my_name == 'fc.0.weight':
	# 		val = torch.from_numpy(tf_val).float()
	# 		val = val.reshape(64, 4, 4, 256).permute(3,1,2,0).reshape(256, 1024)
	# 		# val = val.reshape(256, 64, 4, 4).permute(0,3,2,1).reshape(val.size(0),-1)
	# 		# val = val.reshape(256, 64, 4, 4).permute(0,2,3,1).reshape(val.size(0),-1)
	# 		# val = val.reshape(256, 4, 4, 64).permute(0,2,3,1).reshape(val.size(0),-1)
	# 		return val
	#
	# 	val = super()._convert_tf_val(my_name, param, tf_val)
	#
	# 	return val

	_torch_save_name = 'encoder'
	_tf_var_names = {'conv.0.weight': 'encoder/e1/kernel',
	              'conv.0.bias': 'encoder/e1/bias',
	              'conv.2.weight': 'encoder/e2/kernel',
	              'conv.2.bias': 'encoder/e2/bias',
	              'conv.4.weight': 'encoder/e3/kernel',
	              'conv.4.bias': 'encoder/e3/bias',
	              'conv.6.weight': 'encoder/e4/kernel',
	              'conv.6.bias': 'encoder/e4/bias',
	              'fc.0.weight': 'encoder/e5/kernel',
	              'fc.0.bias': 'encoder/e5/bias',
	              # 'conv.9.weight': 'encoder/e5/kernel',
	              # 'conv.9.bias': 'encoder/e5/bias',
	              'mu_layer.weight': 'encoder/means/kernel',
	                 'mu_layer.bias': 'encoder/means/bias',
	                 'logsigma_layer.weight': 'encoder/log_var/kernel',
	                 'logsigma_layer.bias': 'encoder/log_var/bias',
	                 }



@fig.Component('dislib-dec-arch')
class Disentanglement_lib_Decoder_Arch(learn.Decodable, learn.Optimizable, learn.Function):
	def __init__(self, A, din=None, dout=None, **kwargs):
		
		if din is None:
			din = A.pull('latent-dim', '<>din')
		if dout is None:
			dout = A.pull('out-shape', '<>dout')
		
		C, H, W = dout
		
		flat_shape = 64, H // 2 ** 4, W // 2 ** 4
		flat_dim = int(np.product(flat_shape))
		
		hidden_dim = A.pull('hidden_dim', 256)
		
		fc = nn.Sequential(
			nn.Linear(din, hidden_dim),
			util.get_nonlinearity(A.pull('nonlin', 'relu')),
			nn.Linear(hidden_dim, flat_dim),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
		)
		
		conv = nn.Sequential(
			# models.Reshaper(flat_shape),
			nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity(A.pull('nonlin', 'relu', silent=True)),
			nn.ConvTranspose2d(32, C, kernel_size=4, stride=2, padding=1),
			util.get_nonlinearity('sigmoid')
		)
		
		super().__init__(A, din=din, dout=dout, **kwargs)
		
		self.flat_shape = flat_shape
		
		self.fc = fc
		self.conv = conv
	
	def forward(self, q):
		if isinstance(q, distrib.Normal):
			q = q.rsample()
			
		c = self.fc(q)
		C, H, W = self.flat_shape
		c = c.reshape(-1, H, W, C).permute(0, 3, 1, 2)
		
		return self.conv(c)
	
	def decode(self, q):
		return self(q)

@fig.Component('dislib-dec')
class Disentanglement_lib_Decoder(learn.TensorflowPort, Disentanglement_lib_Decoder_Arch):
	def __init__(self, A, tf_path=unspecified_argument, din=None, dout=None, **kwargs):
		
		if tf_path is unspecified_argument:
			tf_path = self._process_tf_path(A, tf_path=tf_path)
		if tf_path is not None:
			din, dout = 10, (3, 64, 64)
		
		super().__init__(A, tf_path=tf_path, din=din, dout=dout, **kwargs)
	
	# def _convert_tf_val(self, my_name, param, tf_val):
	# 	val = super()._convert_tf_val(my_name, param, tf_val)
	#
	# 	# if my_name == 'conv.0.weight':
	# 	# 	val = val.reshape(256, 4, 4, 64).permute(0, 2, 3, 1).reshape(val.size(0), -1)
	#
	# 	return val
	
	_torch_save_name = 'decoder'
	_tf_var_names = {
		'fc.0.weight': 'decoder/dense/kernel',
		'fc.0.bias': 'decoder/dense/bias',
		'fc.2.weight': 'decoder/dense_1/kernel',
		'fc.2.bias': 'decoder/dense_1/bias',
		
		'conv.0.weight': 'decoder/conv2d_transpose/kernel',
		'conv.0.bias': 'decoder/conv2d_transpose/bias',
		'conv.2.weight': 'decoder/conv2d_transpose_1/kernel',
		'conv.2.bias': 'decoder/conv2d_transpose_1/bias',
		'conv.4.weight': 'decoder/conv2d_transpose_2/kernel',
		'conv.4.bias': 'decoder/conv2d_transpose_2/bias',
		'conv.6.weight': 'decoder/conv2d_transpose_3/kernel',
		'conv.6.bias': 'decoder/conv2d_transpose_3/bias'
		
		# 'conv.1.weight': 'decoder/conv2d_transpose/kernel',
		# 'conv.1.bias': 'decoder/conv2d_transpose/bias',
		# 'conv.3.weight': 'decoder/conv2d_transpose_1/kernel',
		# 'conv.3.bias': 'decoder/conv2d_transpose_1/bias',
		# 'conv.5.weight': 'decoder/conv2d_transpose_2/kernel',
		# 'conv.5.bias': 'decoder/conv2d_transpose_2/bias',
		# 'conv.7.weight': 'decoder/conv2d_transpose_3/kernel',
		# 'conv.7.bias': 'decoder/conv2d_transpose_3/bias'
	}



@fig.Component('tcvae')
class TCVAE(VAE):
	def __init__(self, A, anneal_steps=unspecified_argument,
	             alpha=unspecified_argument, beta=unspecified_argument, gamma=unspecified_argument,
	             **kwargs):

		if anneal_steps is unspecified_argument:
			anneal_steps = A.pull('anneal-steps', 200)

		if alpha is unspecified_argument:
			alpha = A.pull('alpha', 1)
		if beta is unspecified_argument:
			beta = A.pull('beta', 6)
		if gamma is unspecified_argument:
			gamma = A.pull('gamma', 1)

		super().__init__(A, **kwargs)

		self.register_attr('num_iter', 0)
		self.register_hparams(alpha=alpha, beta=beta, gamma=gamma, anneal_steps=anneal_steps)
		self.register_stats('mi-loss', 'tc-loss', 'kld-loss')

		self.register_buffer('_wt_mat', None)


	def log_importance_weight_matrix(self, batch_size, dataset_size=None):
		"""
		Calculates a log importance weight matrix
		Parameters
		from: https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
		----------
		batch_size: int
			number of training images in the batch
		dataset_size: int
		number of training images in the dataset
		"""
		if dataset_size is None:
			dataset_size = self._dataset_size
		N = dataset_size
		M = batch_size - 1
		strat_weight = (N - M) / (N * M)
		W = torch.ones(batch_size, batch_size, device=self.device).mul_(1 / M)
		W.view(-1)[::M + 1] = 1 / N
		W.view(-1)[1::M + 1] = strat_weight
		W[M - 1, 0] = strat_weight
		return W.log().view(batch_size, batch_size, 1)


	def prep(self, order, info=None):
		dataset = info.get_dataset()
		N = len(dataset)
		B = dataset.get_batch_size()
		self._dataset_size = N
		self._wt_mat = self.log_importance_weight_matrix(B, N)

		return super().prep(order, info=info)


	def _reg_step(self, out):
		reg_loss = self.regularize(out.latent, out.latent_samples)
		self.mete('reg-loss', reg_loss)
		out.reg_loss = reg_loss
		return self.reg_wt * reg_loss


	def regularize(self, qdis, q):
		# from https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py#L144


		importance_mat = self._wt_mat if self._wt_mat is None or len(q) == len(self._wt_mat) \
			else self.log_importance_weight_matrix(len(q))
		mat_log_q_z = self.get_prior(1, len(q), self.prior_dim).log_prob(q.unsqueeze(1))

		log_q_zx = qdis.log_prob(q).sum(-1)
		log_p_z = self.get_prior().log_prob(q).sum(-1)
		# log_p_z = mat_log_q_z.diagonal(0,0,1).sum(0)

		mat_log_q_z = mat_log_q_z + importance_mat

		log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
		log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

		mi_loss = (log_q_zx - log_q_z).mean()
		tc_loss = (log_q_z - log_prod_q_z).mean()
		kld_loss = (log_prod_q_z - log_p_z).mean()

		self.mete('mi-loss', mi_loss)
		self.mete('tc-loss', tc_loss)
		self.mete('kld-loss', kld_loss)

		# kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

		if self.training:
			self.num_iter += 1
			anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
		else:
			anneal_rate = 1.

		loss = self.alpha * mi_loss + \
		       self.beta * tc_loss + \
		       anneal_rate * self.gamma * kld_loss
		return loss



@fig.Component('fvae')
class FVAE(VAE):
	def __init__(self, A, discriminator=unspecified_argument, gamma=unspecified_argument, disc_wt=unspecified_argument,
	             **kwargs):

		if discriminator is unspecified_argument:
			discriminator = A.pull('discriminator')

		if gamma is unspecified_argument:
			gamma = A.pull('gamma', 40)

		if disc_wt is unspecified_argument:
			disc_wt = A.pull('disc-wt', 1)
		super().__init__(A, **kwargs)

		self.discriminator = discriminator
		self.register_attr('take_disc_step', False)
		self.register_hparams(gamma=gamma, disc_wt=disc_wt)
		self.register_stats('tc-loss-disc', 'tc-loss', 'kl-loss', 'disc-true', 'disc-false')
	
	
	def permute_latent(self, z):
		"""
		Permutes each of the latent codes in the batch
		:param z: [B x D]
		:return: [B x D]
		"""
		B, D = z.size()
		
		# Returns a shuffled inds for each latent code in the batch
		inds = torch.cat([(D * i) + torch.randperm(D) for i in range(B)])
		return z.view(-1)[inds].view(B, D)
	
	
	def _disc_step(self, out):
		device = self.device
		
		for param in self.discriminator.parameters():
			param.requires_grad = True

		z_perm = self.permute_latent(out.latent_samples.detach())
		
		true_labels = torch.ones(z_perm.size(0), dtype=torch.long,
		                         requires_grad=False).to(device)
		false_labels = torch.zeros(z_perm.size(0), dtype=torch.long,
		                           requires_grad=False).to(device)
		
		D_z_perm = self.discriminator(z_perm)
		D_z = self.discriminator(out.latent.sample().detach())
		D_tc_loss = 0.5 * (F.cross_entropy(D_z, false_labels) +
		                   F.cross_entropy(D_z_perm, true_labels))
		
		self.mete('tc-loss-disc', D_tc_loss)
		self.mete('disc-true', F.softmax(D_z_perm.detach(), 1)[:, 0].mean())
		self.mete('disc-false', F.softmax(D_z.detach(), 1)[:, 0].mean())
		
		for param in self.discriminator.parameters():
			param.requires_grad = False
		
		return self.disc_wt * D_tc_loss
	
	
	def _reg_step(self, out):
		if 'latent' not in out:
			out.latent = self.encode(out.original)
			out.latent_samples = out.latent.rsample() if isinstance(out.latent, util.Distribution) else out.latent

		q = out.latent
		kld_loss = self.regularize(q)
		
		D_z = self.discriminator(out.latent_samples)
		vae_tc_loss = (D_z[:, 0] - D_z[:, 1]).mean()
		
		reg_loss = self.reg_wt * kld_loss + self.gamma * vae_tc_loss
		
		self.mete('kl-loss', kld_loss)
		self.mete('tc-loss', vae_tc_loss)
		self.mete('reg-loss', reg_loss)
		
		disc_loss = self._disc_step(out)
		
		return reg_loss + disc_loss

	
	# def _step(self, batch, out=None):
	#
	# 	if self.train_me() and self.take_disc_step:
	# 		out = self._process_batch(batch, out)
	#
	# 		if self.train_me():
	# 			self.optim.zero_grad()
	#
	# 		loss = self._disc_step(out)
	#
	# 		if self.train_me():
	# 			loss.backward()
	# 			self.optim.step()
	# 		out.reconstruction, out.original = self._viz_rec, self._viz_ori
	# 	else:
	# 		out = super()._step(batch, out=out)
	# 		self._viz_rec, self._viz_ori = out.reconstruction, out.original
	#
	# 	if self.train_me():
	# 		self.take_disc_step ^= True
	# 	return out
		
	
	# def loss_function(self,
	#                   *args,
	#                   **kwargs) -> dict:
	# 	"""
	# 	Computes the VAE loss function.
	# 	KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
	#
	# 	from https://github.com/AntixK/PyTorch-VAE/blob/master/models/fvae.py
	#
	# 	:param args:
	# 	:param kwargs:
	# 	:return:
	# 	"""
	# 	recons = args[0]
	# 	input = args[1]
	# 	mu = args[2]
	# 	log_var = args[3]
	# 	z = args[4]
	#
	# 	kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
	# 	optimizer_idx = kwargs['optimizer_idx']
	#
	# 	# Update the VAE
	# 	if optimizer_idx == 0:
	# 		recons_loss = F.mse_loss(recons, input)
	# 		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
	#
	# 		self.D_z_reserve = self.discriminator(z)
	# 		vae_tc_loss = (self.D_z_reserve[:, 0] - self.D_z_reserve[:, 1]).mean()
	#
	# 		loss = recons_loss + kld_weight * kld_loss + self.gamma * vae_tc_loss
	#
	# 		# print(f' recons: {recons_loss}, kld: {kld_loss}, VAE_TC_loss: {vae_tc_loss}')
	# 		return {'loss': loss,
	# 		        'Reconstruction_Loss': recons_loss,
	# 		        'KLD': -kld_loss,
	# 		        'VAE_TC_Loss': vae_tc_loss}
	#
	# 	# Update the Discriminator
	# 	elif optimizer_idx == 1:
	# 		device = input.device
	# 		true_labels = torch.ones(input.size(0), dtype=torch.long,
	# 		                         requires_grad=False).to(device)
	# 		false_labels = torch.zeros(input.size(0), dtype=torch.long,
	# 		                           requires_grad=False).to(device)
	#
	# 		z = z.detach()  # Detach so that VAE is not trained again
	# 		z_perm = self.permute_latent(z)
	# 		D_z_perm = self.discriminator(z_perm)
	# 		D_tc_loss = 0.5 * (F.cross_entropy(self.D_z_reserve, false_labels) +
	# 		                   F.cross_entropy(D_z_perm, true_labels))
	# 		# print(f'D_TC: {D_tc_loss}')
	# 		return {'loss': D_tc_loss,
	# 		        'D_TC_Loss': D_tc_loss}
	
	# # Discriminator network for the Total Correlation (TC) loss
		# self.discriminator = nn.Sequential(nn.Linear(self.latent_dim, 1000),
		#                                    nn.BatchNorm1d(1000),
		#                                    nn.LeakyReLU(0.2),
		#                                    nn.Linear(1000, 1000),
		#                                    nn.BatchNorm1d(1000),
		#                                    nn.LeakyReLU(0.2),
		#                                    nn.Linear(1000, 1000),
		#                                    nn.BatchNorm1d(1000),
		#                                    nn.LeakyReLU(0.2),
		#                                    nn.Linear(1000, 2))
		# self.D_z_reserve = None


	# def encode(self, input):
	# 	"""
	# 	Encodes the input by passing through the encoder network
	# 	and returns the latent codes.
	# 	:param input: (Tensor) Input tensor to encoder [N x C x H x W]
	# 	:return: (Tensor) List of latent codes
	# 	"""
	# 	result = self.encoder(input)
	# 	result = torch.flatten(result, start_dim=1)
	#
	# 	# Split the result into mu and var components
	# 	# of the latent Gaussian distribution
	# 	mu = self.fc_mu(result)
	# 	log_var = self.fc_var(result)
	#
	# 	return [mu, log_var]


	# def decode(self, z):
	# 	"""
	# 	Maps the given latent codes
	# 	onto the image space.
	# 	:param z: (Tensor) [B x D]
	# 	:return: (Tensor) [B x C x H x W]
	# 	"""
	# 	result = self.decoder_input(z)
	# 	result = result.view(-1, 512, 2, 2)
	# 	result = self.decoder(result)
	# 	result = self.final_layer(result)
	# 	return result


	# def reparameterize(self, mu, logvar):
	# 	"""
	# 	Reparameterization trick to sample from N(mu, var) from
	# 	N(0,1).
	# 	:param mu: (Tensor) Mean of the latent Gaussian [B x D]
	# 	:param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
	# 	:return: (Tensor) [B x D]
	# 	"""
	# 	std = torch.exp(0.5 * logvar)
	# 	eps = torch.randn_like(std)
	# 	return eps * std + mu


	# def forward(self, input, **kwargs):
	# 	mu, log_var = self.encode(input)
	# 	z = self.reparameterize(mu, log_var)
	# 	return [self.decode(z), input, mu, log_var, z]



	# def sample(self,
	#            num_samples: int,
	#            current_device: int, **kwargs):
	# 	"""
	# 	Samples from the latent space and return the corresponding
	# 	image space map.
	# 	:param num_samples: (Int) Number of samples
	# 	:param current_device: (Int) Device to run the model
	# 	:return: (Tensor)
	# 	"""
	# 	z = torch.randn(num_samples,
	# 	                self.latent_dim)
	#
	# 	z = z.to(current_device)
	#
	# 	samples = self.decode(z)
	# 	return samples
	#
	#
	# def generate(self, x, **kwargs):
	# 	"""
	# 	Given an input image x, returns the reconstructed image
	# 	:param x: (Tensor) [B x C x H x W]
	# 	:return: (Tensor) [B x C x H x W]
	# 	"""
	#
	# 	return self.forward(x)[0]

