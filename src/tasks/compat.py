from pathlib import Path
import sys, os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt # TESTING
from torchvision import datasets, transforms
import time

from omnibelt import get_printer
import omnifig as fig

prt = get_printer(__file__)

# try:
	# The code in these files can be found at https://github.com/bits-back/bits-back
	# Also, after the code is copied, you must change the import statements to be relative instead of absolute
from .lossless_compression import util
from .lossless_compression import rans
from .lossless_compression.torch_vae.tvae_beta_binomial import BetaBinomialVAE
from .lossless_compression.torch_vae import tvae_utils
from .lossless_compression.torch_vae.torch_mnist_compress import run_compress
from .lossless_compression.torch_vae.torch_bin_mnist_compress import run_bin_compress

# except ImportError:
# 	prt.warning('Unable to import lossless compression files (find the code at https://github.com/bits-back/bits-back)')

@fig.Script('test-compress')
def _test_compress(A):
	root = Path(__file__).parents[0] / 'lossless_compression'
	if A.pull('bin', False):
		run_bin_compress(root)
	else:
		run_compress(root)


@fig.Script('test-line-compress')
def run_compress(A):
	root = Path(__file__).parents[0] / 'lossless_compression'

	rng = np.random.RandomState(0)
	np.seterr(over='raise')

	prior_precision = 8
	obs_precision = 14
	q_precision = 14

	num_images = 5

	compress_lengths = []

	# latent_dim = 50
	# latent_shape = (1, latent_dim)
	# model = BetaBinomialVAE(hidden_dim=200, latent_dim=latent_dim)
	# model.load_state_dict(
	# 	torch.load(root / 'torch_vae/saved_params/torch_vae_beta_binomial_params',
	# 	           map_location=lambda storage, location: storage))
	# model.eval()
	# rec_net = tvae_utils.torch_fun_to_numpy_fun(model.encode)
	# gen_net = tvae_utils.torch_fun_to_numpy_fun(model.decode)

	run = fig.run('load-run', A)

	model = run.get_model()

	compressor = BitsBackCompressor(encoder=model, decoder=model,
	                                obs_precision=obs_precision, q_precision=q_precision,
	                                prior_precision=prior_precision)

	# latent_dim = compressor.start_seed_len
	# latent_shape = (1, latent_dim)
	#
	# rec_net = tvae_utils.torch_fun_to_numpy_fun(compressor._wrapped_encode)
	# gen_net = tvae_utils.torch_fun_to_numpy_fun(compressor._wrapped_decode)
	#
	#
	# obs_append = tvae_utils.beta_binomial_obs_append(255, obs_precision)
	# obs_pop = tvae_utils.beta_binomial_obs_pop(255, obs_precision)

	# vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append,
	#                              prior_precision, q_precision)
	# vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
	#                        prior_precision, q_precision)

	# load some mnist images
	# mnist = datasets.MNIST('data/mnist', train=False, download=True,
	#                        transform=transforms.Compose([transforms.ToTensor()]))
	# images = mnist.test_data[:num_images]
	#
	# images = F.interpolate(images.float().unsqueeze(1), size=(32,32), mode='bilinear').round()
	#

	dataset = run.get_dataset()
	images = dataset.to_loader(infinite=True).demand(num_images)[0].mul(255).round().cpu().contiguous()

	images = [image.float().view(1, -1) for image in images]

	# randomly generate some 'other' bits
	# other_bits = rng.randint(low=1 << 16, high=1 << 31, size=latent_dim, dtype=np.uint32)
	# state = rans.unflatten(other_bits)


	# state = compressor.generate_seed_state()
	# other_bits = rans.flatten(state)

	print_interval = 10
	encode_start_time = time.time()
	data = compressor.compress(images)
	# state = compressor.compress_append(images, state=state)
	# for i, image in enumerate(images):
	# 	state = vae_append(state, image)
	#
	# 	if not i % print_interval:
	# 		print('Encoded {}'.format(i))
	#
	# 	compressed_length = 32 * (len(rans.flatten(state)) - len(other_bits)) / (i + 1)
	# 	compress_lengths.append(compressed_length)

	print('\nAll encoded in {:.2f}s'.format(time.time() - encode_start_time))

	# compressed_message = rans.flatten(state)

	# compressed_bits = 32 * (len(compressed_message) - len(other_bits))
	compressed_bits = len(data) * 8
	print("Used " + str(compressed_bits) + " bits.")
	print('This is {:.2f} bits per pixel'.format(compressed_bits
	                                             / (num_images * images[0].size(-1))))

	if not os.path.exists('results'):
		os.mkdir('results')
	np.savetxt('compressed_lengths_cts', np.array(compress_lengths))

	# state = rans.unflatten(compressed_message)
	decode_start_time = time.time()

	recs = compressor.decompress(data)
	# state, recs = compressor.partial_decompress(state)

	# for n in range(len(images)):
	# 	state, image_ = vae_pop(state)
	# 	original_image = images[len(images) - n - 1].numpy()
	# 	np.testing.assert_allclose(original_image, image_)
	#
	# 	if not n % print_interval:
	# 		print('Decoded {}'.format(n))

	print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))

	# recovered_bits = rans.flatten(state)
	# assert all(other_bits == recovered_bits)
	print('done')


class BitsBackCompressor:
	def __init__(self, encoder, decoder, latent_shape=None, seed=None,
	             output_distribution='beta', beta_confidence=1000,
	             obs_precision=14, q_precision=14, prior_precision=8):
		if latent_shape is None:
			latent_shape = getattr(encoder, 'latent_dim', encoder.dout)

			if isinstance(latent_shape, int):
				latent_shape = 1, latent_shape
		latent_dim = np.product(latent_shape).item()

		# self.state = None
		self.start_seed_len = latent_dim
		self.rng = np.random.RandomState(seed)

		self.encoder = encoder
		self.decoder = decoder

		rec_net = tvae_utils.torch_fun_to_numpy_fun(self._wrapped_encode)
		gen_net = tvae_utils.torch_fun_to_numpy_fun(self._wrapped_decode)

		if output_distribution == 'beta':
			obs_append = tvae_utils.beta_binomial_obs_append(255, obs_precision)
			obs_pop = tvae_utils.beta_binomial_obs_pop(255, obs_precision)
		else:
			obs_append = tvae_utils.bernoulli_obs_append(obs_precision)
			obs_pop = tvae_utils.bernoulli_obs_pop(obs_precision)
		self._output_distribution = output_distribution
		# self._beta_variance = 1 - beta_confidence
		assert beta_confidence > 1
		self._beta_confidence = beta_confidence

		self.vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append,
		                             prior_precision, q_precision)
		self.vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
		                       prior_precision, q_precision)


	def _wrapped_encode(self, x):
		x = x.to(self.encoder.get_device()).div(255).view(-1, *self.encoder.din)#.unsqueeze(0)
		with torch.no_grad():
			z = self.encoder.encode(x)
		return z.loc.cpu(), z.scale.cpu()


	def _wrapped_decode(self, z):
		z = z.to(self.decoder.get_device()).unsqueeze(0)
		with torch.no_grad():
			x = self.decoder.decode(z)
		x = x.cpu().view(z.size(0), -1)
		if self._output_distribution == 'beta':
			x = self._compute_beta_params(x)
		return x


	def _compute_beta_params(self, x):
		# x = x.clamp(min=1e-8, max=1-1e-8)
		a = x.mul(self._beta_confidence).clamp(min=1e-8)
		b = (1-x).mul(self._beta_confidence).clamp(min=1e-8)
		return a,b


	def generate_seed_state(self, N=None):
		if N is None:
			N = self.start_seed_len
		stream = self.rng.randint(low=1 << 16, high=1 << 31, size=N, dtype=np.uint32)
		return rans.unflatten(stream)


	def state_to_bytes(self, state):
		nums = rans.flatten(state)
		return bytes(bytearray(nums))


	def bytes_to_state(self, data):
		nums = np.frombuffer(data, dtype=np.uint32)

		# state = tuple(np.int32(int.from_bytes(x, byteorder='little', signed=True)) for x in data)
		return rans.unflatten(nums)


	# def set_state(self, state=None):
	# 	if state is None:
	# 		state = self.generate_seed_state()
	# 	self.state = state
	# 	return state


	def count_bits(self, state):
		# if state is None:
		# 	state = self.state
		# return 8 * (len(rans.flatten(state).tobytes()) - 4*self.start_seed_len)
		return 32 * (len(rans.flatten(state)) - self.start_seed_len)


	def compress_append(self, images, state=None):
		if state is None:
			state = self.generate_seed_state()
			# if self.state is None:
			# 	state = self.generate_seed_state()
		# counts = []
		for image in images:
			state = self.vae_append(state, image.cpu().float().unsqueeze(0)) #.mul(255).round()
			# counts.append(self.count_bits(state))
		# self.state = state
		return state#, counts


	def partial_decompress(self, state, N=None):
		imgs = []
		while (N is not None and N > 0) or (N is None and len(rans.flatten(state)) > self.start_seed_len):
			state, img = self.vae_pop(state)
			imgs.append(img)
			if N is not None:
				N -= 1
		# self.state = state
		imgs = torch.stack(imgs[::-1]).round().byte().view(-1, *self.decoder.dout).to(self.decoder.get_device())#.div(255)
		return state, imgs


	def compress(self, images, state=None):
		state = self.compress_append(images, state)
		return self.state_to_bytes(state)


	def decompress(self, data):
		state = self.bytes_to_state(data)
		return self.partial_decompress(state)[-1]




class GroundTruthData(object):
	"""Abstract class for data sets that are two-step generative models."""

	@property
	def num_factors(self):
		raise NotImplementedError()

	@property
	def factors_num_values(self):
		raise NotImplementedError()

	@property
	def observation_shape(self):
		raise NotImplementedError()



class DisentanglementDatasetCompat(GroundTruthData):
	def __init__(self, dataset):
		dataset.switch_to('full')
		self.base = dataset

	@property
	def num_factors(self):
		return len(self.base.get_target_space())

	@property
	def factors_num_values(self):
		return [dim.expanded_len() for dim in self.base.get_target_space()]

	@property
	def observation_shape(self):
		return self.base.din


	def sample_factors(self, num, random_state, to_numpy=True):
		"""Sample a batch of factors Y."""
		factors = self.base.generate_labels(num, seed=int.from_bytes(random_state.bytes(4),
		                                                           byteorder='big', signed=True))
		if to_numpy:
			factors = factors.numpy()
		return factors


	def sample_observations_from_factors(self, factors, random_state, to_numpy=True):
		"""Sample a batch of observations X given a batch of factors Y."""
		obs = self.base.true_generative_process(factors, seed=int.from_bytes(random_state.bytes(4),
		                                                           byteorder='big', signed=True))
		if to_numpy:
			obs = obs.numpy()
		return obs



	def sample(self, num, random_state, to_numpy=True):
		"""Sample a batch of factors Y and observations X."""
		factors = self.sample_factors(num, random_state, to_numpy=False)
		obs = self.sample_observations_from_factors(factors, random_state, to_numpy=False)
		if to_numpy:
			factors = factors.numpy()
			obs = obs.numpy()
		return factors, obs


	def sample_observations(self, num, random_state, to_numpy=True):
		"""Sample a batch of observations X."""
		return self.sample(num, random_state, to_numpy=to_numpy)[1]


