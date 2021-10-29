
import omnifig as fig
import random
import torch
import zlib
from omnibelt import unspecified_argument
#
# from omnilearn.op.datasets import MPI3D, Shapes3D, CelebA
from omnilearn.data import register_dataset, JointFactorSampler, InterventionSampler, DatasetBase, Batchable, Deviced
from omnilearn.util import Configurable, InitWall, distributions as distrib
from omnilearn.data import Supervised, Dataset, Batchable, Deviced, Observation, DatasetWrapper


@DatasetWrapper('encoded')
class Encoded(Batchable, Observation):
	def __init__(self, A, encoder=unspecified_argument, sample_best=None, batch_size=None, auto_process=None,
	             pbar=unspecified_argument, **kwargs):

		if encoder is unspecified_argument:
			encoder = A.pull('encoder', '<>model', None, ref=True)

		if sample_best is None:
			sample_best = A.pull('sample-best', True)

		if batch_size is None:
			batch_size = A.pull('batch-size', 64)

		if pbar is unspecified_argument:
			pbar = A.pull('progress-bar', None)

		if auto_process is None:
			auto_process = A.pull('auto-process', True)

		super().__init__(A, **kwargs)
		self.__activate__(encoder=encoder, sample_best=sample_best, batch_size=batch_size, auto_process=auto_process,
		                  pbar=pbar)

	def __activate__(self, encoder=None, sample_best=True, batch_size=64, auto_process=True, pbar=None):
		self.encoder = encoder
		self._sample_best = sample_best
		self._auto_process = auto_process
		self._observations_encoded = False
		self._batch_size = batch_size
		self._pbar = pbar
		self._skip_encoder = False

		if self.encoder is not None:
			self.set_encoder(encoder)

	def allow_batched_get(self):
		return self._observations_encoded

	def collate(self, samples):
		if self.allow_batched_get():
			return super().collate(samples)
		return super(Batchable, self).collate(samples)

	def set_encoder(self, encoder):
		self.encoder = encoder
		self.din = getattr(self.encoder, 'latent_dim', getattr(self.encoder, 'dout', None))
		if self._auto_process:
			self._process_observations()

	def _process_observations(self):  # TODO: replace with "process_in_batches"
		loader = self.to_loader(sample_format='observations', batch_size=self._batch_size, shuffle=False)
		self._skip_encoder = True

		if self._pbar is not None:
			self._pbar.set_description('Encoding observations')
			loader = self._pbar(loader)

		Q = []
		with torch.no_grad():
			for x in loader:
				# print(len(x))
				q = self._encode_observation(x)
				Q.append(q.cpu())
		Q = torch.cat(Q)

		self._observations_encoded = True
		self._available_data_keys['observations'] = None
		self._skip_encoder = False
		if isinstance(self, Deviced):
			self.register_buffer('observations', Q)
		else:
			self.register_data('observations', data=Q)

	# def get_features(self, idx=None): # TODO: instead of replacing observations, just link them to "features"
	# 	pass

	# @DatasetWrapper.condition(Observation)
	def get_observations(self, idx=None):
		if self._observations_encoded:
			return self.observations if idx is None else self.observations[idx]
		obs = super().get_observations(idx=idx)
		if self._skip_encoder:
			return obs
		if self.encoder is not None:
			obs = self._encode_observation(obs)
		return obs

	def _encode_observation(self, x, **kwargs):
		# print(x.shape)
		encode = getattr(self.encoder, 'encode', self.encoder)
		z = encode(x.to(self.encoder.get_device()), **kwargs)
		if isinstance(z, distrib.Distribution):
			z = z.bsample() if self._sample_best else z.rsample()
		return z


# def __getitem__(self, item):
# 	if self._observations_encoded and not isinstance(self, Batchable):
# 		return self.observations
# 	x, *other = super().__getitem__(item)
# 	if not self._observations_encoded:
# 		x = self._encode_observation(x)
# 	return x, *other


class SimpleVectorDataset(Batchable, DatasetBase):

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

from .scm import SCM_Simul
from .scm.data import erdos_renyi

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

