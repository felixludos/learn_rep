
<<<<<<< HEAD
import omnifig as fig
import random
import torch
import zlib
from omnibelt import unspecified_argument

from omnilearn.op.datasets import MPI3D, Shapes3D, CelebA
from omnilearn.data import register_dataset, JointFactorSampler, InterventionSampler, DatasetBase, Batchable, Deviced
from omnilearn.util import Configurable, InitWall


class SimpleVectorDataset(Deviced, Batchable, DatasetBase):
	
	def __init__(self, A, num_samples=None, num_nodes=None, out_dim=None, **kwargs):
		
		device = A.pull('device')
		seed = A.pull('seed')
		mode = A.pull('mode', 'train')
		# if mode == 'test':
		# 	seed += 1
		# seed += zlib.adler32(mode.encode()) # deterministically change seed depending on train/test split
		# seed %= 2**32
		print(f'Vector dataset seed: {seed}')

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
		self.seed = seed
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


@register_dataset('random-net')
class RandomNetDataset(SimpleVectorDataset):
	def __init__(self, A, net=unspecified_argument, **kwargs):
		seed = A.pull('seed')
		num_nodes = A.pull('num-nodes', 8)
		
		if net is unspecified_argument:
			torch.manual_seed(seed) # should not change in testset
			net = A.pull('net', None)
			if net is not None:
				for param in net.parameters():
					param.requires_grad = False
		
		super().__init__(A, num_nodes=num_nodes, out_dim=num_nodes if net is None else net.dout, **kwargs)
		
		if net is not None:
			net.to(self.get_device())
		self.net = net

	def _process_prior(self, prior):
		if self.net is None:
			return prior
		return self.net(prior).detach()
=======
# import omnifig as fig
# import random
# import torch
# import zlib
# from omnibelt import unspecified_argument
#
# from omnilearn.op.datasets import MPI3D, Shapes3D, CelebA
# from omnilearn.data import register_dataset, JointFactorSampler, InterventionSampler, Dataset, Batchable, Deviced
# from omnilearn.util import Configurable, InitWall


# class SimpleVectorDataset(Deviced, Batchable, Dataset):
#
# 	def __init__(self, A, num_samples=None, num_nodes=None, out_dim=unspecified_argument,
# 	             dim=None, dout=None, **kwargs):
#
# 		device = A.pull('device')
# 		seed = A.pull('seed')
# 		mode = A.pull('mode', 'train')
# 		# if mode == 'test':
# 		# 	seed += 1
# 		# seed += zlib.adler32(mode.encode()) # deterministically change seed depending on train/test split
# 		# seed %= 2**32
# 		print(f'Vector dataset seed: {seed}')
#
# 		labeled = A.pull('labeled', False)
#
# 		if num_nodes is None:
# 			num_nodes = A.pull('num-nodes', 8)
# 		if num_samples is None:
# 			num_samples = A.pull('num-samples', 100000)
#
# 		if out_dim is unspecified_argument:
# 			out_dim = A.pull('out-dim', 128)
#
# 		super().__init__(A, din=out_dim, dout=num_nodes if labeled else out_dim, device=device, **kwargs)
#
# 		self.num_samples = num_samples
# 		self.num_nodes = num_nodes
#
# 		self.labeled = labeled
# 		self.seed = seed
# 		self.rng = torch.Generator(device=device).manual_seed(seed)
#
# 		self.prior = None
# 		self.samples = None
#
# 	def get_prior(self):
# 		if self.prior is None:
# 			self.prior = torch.randn(self.num_samples, self.num_nodes, generator=self.rng)
# 		return self.prior
#
# 	def _process_prior(self, prior):
# 		raise NotImplementedError
#
# 	def process_prior(self, prior=None):
# 		if prior is None:
# 			prior = self.get_prior()
# 		return self._process_prior(prior)
#
# 	def __len__(self):
# 		return self.num_samples
#
# 	def __getitem__(self, item):
# 		if self.samples is None:
# 			self.samples = self.process_prior()
#
# 		if self.labeled:
# 			return self.samples[item], self.prior[item]
# 		return self.samples[item]


# @register_dataset('random-net')
# class RandomNetDataset(SimpleVectorDataset):
# 	def __init__(self, A, net=unspecified_argument, **kwargs):
# 		seed = A.pull('seed')
# 		num_nodes = A.pull('num-nodes', 8)
#
# 		if net is unspecified_argument:
# 			torch.manual_seed(seed) # should not change in testset
# 			net = A.pull('net', None)
# 			if net is not None:
# 				for param in net.parameters():
# 					param.requires_grad = False
#
# 		super().__init__(A, num_nodes=num_nodes, out_dim=num_nodes if net is None else net.dout, **kwargs)
#
# 		if net is not None:
# 			net.to(self.get_device())
# 		self.net = net
#
# 	def _process_prior(self, prior):
# 		if self.net is None:
# 			return prior
# 		return self.net(prior).detach()
>>>>>>> 7552c6b695501039e3693d5febbccc2815f217c1


from .scm import SCM_Simul
from .scm.data import erdos_renyi

<<<<<<< HEAD
@register_dataset('random-scm')
class RandomSCMDataset(RandomNetDataset):
	def __init__(self, A, **kwargs):
		super().__init__(A, **kwargs)
		
		self.er = erdos_renyi.ER(self.num_nodes, num_samples=self.num_samples, seed=self.seed)
		self.prior = self.er.samples.to(self.get_device())

	def get_factor_order(self):
		return list(map(str,range(self.num_nodes)))
	
	def get_adjacency_matrix(self):
		return self.er.adjacency_matrix.copy()



@fig.Component('scm-interventions')
class SCMSampler(InterventionSampler):
	
	def __init__(self, A, batch_size=None, **kwargs):
		if batch_size is None:
			batch_size = A.pull('batch_size', 128)
		
		super().__init__(A, **kwargs)
		
		self.batch_size = batch_size
	
	def __len__(self):
		return self.dataset.num_nodes
	
	@property
	def factors_num_values(self):
		return [None]*self.dataset.num_nodes
	
	def intervention(self, idx=None, B=None, val=None):
		if B is None:
			B = self.batch_size
		prior = self.dataset.er.intervene(B, idx, val)[0]
		return self.dataset.process_prior(prior)
	
	def full_intervention(self, idx=None, B=None, vals=None):
		if B is None:
			B = self.batch_size
		if idx is None:
			idx = random.randint(0, self.num_factors-1)
		if vals is not None:
			vals = list(vals.t())
			del vals[idx]
		
		inds = list(range(self.num_factors))
		del inds[idx]
		
		prior = self.dataset.er.multi_intervene(B, inds, vals)[0]
		return self.dataset.process_prior(prior)
		
	
	pass
=======
# @Dataset('linear-scm')
# class LinearSCMDataset(RandomNetDataset):
# 	def __init__(self, A, **kwargs):
# 		super().__init__(A, **kwargs)
#
# 		self.er = erdos_renyi.ER(self.num_nodes, num_samples=self.num_samples, seed=self.seed)
# 		self.prior = self.er.samples.to(self.get_device())
#
# 	def get_factor_order(self):
# 		return list(map(str,range(self.num_nodes)))
#
# 	def get_adjacency_matrix(self):
# 		return self.er.adjacency_matrix.copy()


# @Dataset('complex-scm')
# class ComplexSCMDataset(RandomNetDataset)


# @fig.Component('scm-interventions')
# class SCMSampler(InterventionSampler):
#
# 	def __init__(self, A, batch_size=None, **kwargs):
# 		if batch_size is None:
# 			batch_size = A.pull('batch_size', 128)
#
# 		super().__init__(A, **kwargs)
#
# 		self.batch_size = batch_size
#
# 	def __len__(self):
# 		return self.dataset.num_nodes
#
# 	@property
# 	def factors_num_values(self):
# 		return [None]*self.dataset.num_nodes
#
# 	def intervention(self, idx=None, B=None, val=None):
# 		if B is None:
# 			B = self.batch_size
# 		prior = self.dataset.er.intervene(B, idx, val)[0]
# 		return self.dataset.process_prior(prior)
#
# 	def full_intervention(self, idx=None, B=None, vals=None):
# 		if B is None:
# 			B = self.batch_size
# 		if idx is None:
# 			idx = random.randint(0, self.num_factors-1)
# 		if vals is not None:
# 			vals = list(vals.t())
# 			del vals[idx]
#
# 		inds = list(range(self.num_factors))
# 		del inds[idx]
#
# 		prior = self.dataset.er.multi_intervene(B, inds, vals)[0]
# 		return self.dataset.process_prior(prior)
#
#
# 	pass
>>>>>>> 7552c6b695501039e3693d5febbccc2815f217c1

