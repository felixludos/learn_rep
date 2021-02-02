
import omnifig as fig

import torch

from omnibelt import unspecified_argument

from foundation.op.datasets import MPI3D, Shapes3D, CelebA
from foundation.data import Dataset, JointFactorSampler, DatasetBase, Batchable, Deviced

@Dataset('full-mpi3d')
class Full_MPI3D(MPI3D):
	
	def __init__(self, A, sort=None, **kwargs):
		
		if sort is None:
			sort = A.pull('sort', True)
		
		super().__init__(A, mode='train', **kwargs)
		
		img1, idx1 = self.images, self.indices
		
		with A.silenced():
			super().__init__(A, mode='test', **kwargs)
		
		self.images = torch.cat([img1, self.images])
		self.indices = torch.cat([idx1, self.indices])
	
		self.order = self.indices.argsort() if sort else None
	
	def __getitem__(self, item):
		if self.order is not None:
			item = self.order[item]
		return super().__getitem__(item)
		

class SimpleVectorDataset(Deviced, Batchable, DatasetBase):
	
	def __init__(self, A, num_samples=None, num_nodes=None, out_dim=None, **kwargs):
		
		device = A.pull('device')
		seed = A.pull('seed')
		mode = A.pull('mode', 'train')
		seed += hash(mode) # deterministically change seed depending on train/test split

		labeled = A.pull('labeled', False)

		if num_nodes is None:
			num_nodes = A.pull('num-nodes', 8)
		if num_samples is None:
			num_samples = A.pull('num-samples', 100000)

		if out_dim is unspecified_argument:
			out_dim = A.pull('out-dim', 128)
		
		super().__init__(A, din=out_dim, dout=num_nodes if labeled else out_dim, device=device, **kwargs)

		self.num_samples = num_samples
		self.num_nodes = num_nodes
		
		self.labeled = labeled
		self.rng = torch.Generator(device=device).manual_seed(seed)
		
		self.prior = None
		self.samples = None
		
	def get_prior(self):
		if self.prior is None:
			self.prior = torch.randn(self.num_samples, self.num_nodes, generator=self.rng)
		return self.prior

	def _process_prior(self, prior):
		raise NotImplementedError

	def process_prior(self, prior=None):
		if prior is None:
			prior = self.get_prior()
		return self._process_prior(prior)

	def __len__(self):
		return self.num_samples

	def __getitem__(self, item):
		if self.samples is None:
			self.samples = self.process_prior()
			
		if self.labeled:
			return self.samples[item], self.prior[item]
		return self.samples[item]


@Dataset('random-net')
class RandomNetDataset(SimpleVectorDataset):
	def __init__(self, A, net=None, **kwargs):

		num_nodes = A.pull('num-nodes', 8)
		
		if net is None:
			net = A.pull('net')
		for param in net.parameters():
			param.requires_grad = False
		
		super().__init__(A, num_nodes=num_nodes, out_dim=net.dout, **kwargs)
		
		net.to(self.get_device())
		self.net = net

	def _process_prior(self, prior):
		return self.net(prior).detach()





