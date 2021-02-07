import numpy as np
import torch

PRESETS = ['chain', 'collider','fork', 'random']
NOISE_TYPES = ['gaussian', 'isotropic-gaussian', 'exponential', 'gumbel']
VARIABLE_TYPES = ['gaussian', 'non-gaussian', 'categorical']

class Generator(torch.utils.data.Dataset):
	def __init__(self, num_nodes, num_edges, noise_type, num_samples, seed = None):
		self.num_nodes = num_nodes
		self.num_edges = num_edges
		assert noise_type in NOISE_TYPES, 'Noise types must correspond to {} but got {}'.format(NOISE_TYPES, noise_type)
		self.noise_type = noise_type
		self.num_samples = num_samples
		if seed is not None:
			self.reseed(seed)

		self.sample_weights()
		self.build_graph()
	
	def reseed(self, seed = None):
		torch.manual_seed(seed)
		np.random.seed(seed)

	def __getitem__(self, index):
		raise NotImplementedError

	def build_graph(self):
		raise NotImplementedError

	def sample_weights(self):
		"""Sample the edge weights"""
		dist = torch.distributions.uniform.Uniform(-5, 5)
		self.weights = torch.zeros(self.num_edges)
		for k in range(self.num_edges):
			sample = 0.
			while sample > -0.5 and sample < 0.5:
				sample = dist.sample()
				self.weights[k] = sample

		print(self.weights)
	
	def __len__(self):
		return self.num_samples


