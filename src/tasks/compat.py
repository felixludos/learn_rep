
import numpy as np
import torch

from omnibelt import get_printer

prt = get_printer(__file__)

try:
	# The code in these files can be found at https://github.com/bits-back/bits-back
	from .lossless_compression import util
	from .lossless_compression import rans
	from .lossless_compression.torch_vae.tvae_beta_binomial import BetaBinomialVAE
	from .lossless_compression.torch_vae import tvae_utils

except ImportError:
	prt.warning('Unable to import lossless compression files (find the code at https://github.com/bits-back/bits-back)')


class BitsBackCompressor:
	def __init__(self, encoder, decoder, latent_shape=None, seed=None,
	             obs_precision=14, q_precision=14, prior_precision=8):
		if latent_shape is None:
			latent_shape = getattr(encoder, 'latent_dim', encoder.dout)

			if isinstance(latent_shape, int):
				latent_shape = latent_shape,
		latent_dim = np.product(latent_shape).item()

		self.state = None
		self.start_seed_len = latent_dim
		self.rng = np.random.RandomState(seed)

		rec_net = tvae_utils.torch_fun_to_numpy_fun(getattr(encoder, 'encode', encoder))
		gen_net = tvae_utils.torch_fun_to_numpy_fun(getattr(decoder, 'decode', encoder))

		obs_append = tvae_utils.beta_binomial_obs_append(255, obs_precision)
		obs_pop = tvae_utils.beta_binomial_obs_pop(255, obs_precision)

		self.vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append,
		                             prior_precision, q_precision)
		self.vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
		                       prior_precision, q_precision)


	def generate_seed_state(self, N=None):
		if N is None:
			N = self.start_seed_len
		return self.rng.randint(low=1 << 16, high=1 << 31, size=N, dtype=np.uint32)


	def state_to_bytes(self, state):
		return bytes(bytearray(rans.flatten(state)))


	def bytes_to_state(self, data):
		return tuple(np.int32(int.from_bytes(x, byteorder='little', signed=True)) for x in data)


	def set_state(self, state):
		self.state = state


	def count_bits(self, state=None):
		if state is None:
			state = self.state
		return 32 * (len(rans.flatten(state)) - self.start_seed_len)


	def compress_append(self, images, state=None):
		if state is None:
			state = self.generate_seed_state()
			# if self.state is None:
			# 	state = self.generate_seed_state()
		for image in images:
			state = self.vae_append(state, image)
		# self.state = state
		return state


	def partial_decompress(self, state, N=None):
		imgs = []
		while (N is not None and N > 0) or (N is None and len(rans.flatten(state)) > self.start_seed_len):
			state, img = self.vae_pop(state)
			imgs.append(img)
			if N is not None:
				N -= 1
		# self.state = state
		imgs = torch.stack(imgs)
		return state, imgs


	def compress(self, images):
		self.state = self.compress_append(images, self.state)
		return self.state_to_bytes(self.state)


	def decompress(self, data):
		self.state = self.bytes_to_state(data)
		return self.partial_decompress(self.state)[-1]


