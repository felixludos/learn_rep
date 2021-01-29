import os
from . import ground_truth_data
from . import utils
import numpy as np
import PIL
from six.moves import range
import h5py as hf

class Shapes3D(ground_truth_data.GroundTruthData):
	"""DSprites dataset.
	The data set was originally introduced in "beta-VAE: Learning Basic Visual
	Concepts with a Constrained Variational Framework" and can be downloaded from
	https://github.com/deepmind/dsprites-dataset.
	The ground-truth factors of variation are (in the default setting):
	0 - shape (3 different values)
	1 - scale (6 different values)
	2 - orientation (40 different values)
	3 - position x (32 different values)
	4 - position y (32 different values)
	"""

	def __init__(self, data_file, latent_factor_indices=None):
		# By default, all factors (including shape) are considered ground truth
		# factors.
		with hf.File(os.path.join('mean_representations/', data_file),'r') as b:
			self.images = np.array(b['embeddings'][()])
			b.close()
		self.data_shape = list(self.images.shape[1:])
		self.factor_sizes = [10, 10, 10, 8, 4, 15]
		self.latent_factor_indices = list(range(6))
		self.num_total_factors = 6
		self.state_space = utils.SplitDiscreteStateSpace(self.factor_sizes,
		                                                self.latent_factor_indices)
		self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
		    self.factor_sizes)

	@property
	def num_factors(self):
		return self.state_space.num_latent_factors

	@property
	def factors_num_values(self):
		return self.factor_sizes

	@property
	def observation_shape(self):
		return self.data_shape

	def sample_factors(self, num, random_state):
		"""Sample a batch of factors Y."""
		return self.state_space.sample_latent_factors(num, random_state)

	def sample_observations_from_factors(self, factors, random_state):
		all_factors = self.state_space.sample_all_factors(factors, random_state)
		indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
		return self.images[indices]