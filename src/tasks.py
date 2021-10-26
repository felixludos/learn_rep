from pathlib import Path

from omnibelt import unspecified_argument, get_printer
import omnifig as fig

import numpy as np

import gym
from gym.spaces import Discrete, MultiDiscrete

import torch
# from torch import distributions as distrib
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from omnilearn import util
from omnilearn.util import distributions as distrib
from omnilearn.models import get_loss_type
from omnilearn.op import get_save_dir, framework as fm#, scikit as sk
from omnilearn.data import Supervised, Dataset, Batchable, Deviced, Observation, DatasetWrapper, DataLike

prt = get_printer(__file__)


@fig.Script('test-estimator')
def _test_estimator(A):

	seed = A.pull('seed', util.gen_random_seed())

	dataset = fig.run('load-data', A)
	# dataset.register_wrapper('subset', kwargs={'num': 100, 'update_data': False})

	# model = fig.run('load-model', A)
	# dataset.set_encoder(model)

	space = dataset.get_target_space()
	print(space)

	builder = A.pull('estimator')
	est = builder.build(dataset)

	# obs = dataset.get('observations')
	# print(obs.shape, )

	# out = est.fit(dataset)

	# pred = est.predict(obs)
	# y = dataset.get('targets')

	dataset.switch_to('val')

	out = est.compute(dataset=dataset)

	# print(pred[:4])
	# print(y[:4])

	print(out.keys())

	pass


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


	def _process_observations(self): # TODO: replace with "process_in_batches"
		loader = self.to_loader(format='observations', batch_size=self._batch_size, shuffle=False)
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



@fig.Component('estimator-builder')
class EstimatorBuilder(util.Builder):
	def __init__(self, A, use_mechanisms=None, **kwargs):
		A.push('default-regressor', 'gbt-regressor', overwrite=False)
		A.push('default-classifier', 'gbt-classifier', overwrite=False)
		super().__init__(A, **kwargs)
		self._use_mechanisms = use_mechanisms


	def _build(self, config, space):
		if isinstance(space, DataLike):
			space = space.get_target_space()

		estimator = None

		if isinstance(space, util.JointSpace):
			config.push('joint._type', 'joint-estimator', overwrite=False, silent=True)
			estimator = config.pull('joint')
			estimator.include_estimators(*[self._build(config, dim) for dim in space])

		elif isinstance(space, util.PeriodicDim):
			config.push('periodic._type', '<>default-regressor', overwrite=False, silent=True)
			config.push('periodic._mod.periodized', 1, overwrite=False, silent=True)
			key = 'periodic'

		elif isinstance(space, util.DenseDim):
			config.push('continuous._type', '<>default-regressor', overwrite=False, silent=True)
			key = 'continuous'

		elif isinstance(space, util.CategoricalDim):
			config.push('categorical._type', '<>default-classifier', overwrite=False, silent=True)
			key = 'categorical'

		if estimator is None:
			estimator = config.pull(key)
			estimator.register_out_space(space)
		return estimator



class Task(fm.Computable):
	def __init__(self, dataset, **kwargs):
		super().__init__(**kwargs)
		self.dataset = dataset

	# def compute(self): # compute() should have not args
	# 	pass

	def _compute(self, dataset=unspecified_argument, **kwargs):
		if dataset is not unspecified_argument:
			self.dataset = dataset
		return self._run(**kwargs)


	def _run(self, out=None):
		if out is None:
			out = util.TensorDict()
		return out



class TaskC(fig.Configurable, Task):
	def __init__(self, A, dataset=unspecified_argument, **kwargs):
		if dataset is unspecified_argument:
			dataset = A.pull('dataset', None, ref=True)
		super().__init__(A, dataset=dataset, **kwargs)



class _EncoderTask(Task):
	def __init__(self, encoder=None, **kwargs):
		super().__init__(**kwargs)
		self.encoder = encoder


	def _run(self, out=None):
		out = super()._run(out=out)
		if self.encoder is not None:
			self.dataset.register_wrapper('encoded', encoder=self.encoder)
		return out


	def _compute(self, encoder=unspecified_argument, **kwargs):
		if encoder is not unspecified_argument:
			self.encoder = encoder
		return super()._compute(**kwargs)



class _EncoderTaskC(TaskC, _EncoderTask):
	def __init__(self, A, encoder=unspecified_argument, **kwargs):
		if encoder is unspecified_argument:
			encoder = A.pull('encoder', None, ref=True)
		super().__init__(A, encoder=encoder, **kwargs)



class _DecoderTask(Task):
	def __init__(self, decoder=None, **kwargs):
		super().__init__(**kwargs)
		self.decoder = decoder


	def _compute(self, decoder=unspecified_argument, **kwargs):
		if decoder is not unspecified_argument:
			self.decoder = decoder
		return super()._compute(**kwargs)



class _DecoderTaskC(TaskC, _DecoderTask):
	def __init__(self, A, decoder=unspecified_argument, **kwargs):
		if decoder is unspecified_argument:
			decoder = A.pull('decoder', None, ref=True)
		super().__init__(A, decoder=decoder, **kwargs)



class _GeneratorTask(Task):
	def __init__(self, generator=None, **kwargs):
		super().__init__(**kwargs)
		self.generator = generator


	def _compute(self, generator=unspecified_argument, **kwargs):
		if generator is not unspecified_argument:
			self.generator = generator
		return super()._compute(**kwargs)



class _GeneratorTaskC(TaskC, _EncoderTask):
	def __init__(self, A, generator=unspecified_argument, **kwargs):
		if generator is unspecified_argument:
			generator = A.pull('generator', None, ref=True)
		super().__init__(A, generator=generator, **kwargs)



class InferenceTask(_EncoderTask):
	def __init__(self, estimator=None, estimator_builder=None, **kwargs):
		if estimator is not None:
			estimator_builder = None
		super().__init__(**kwargs)
		self.estimator = estimator
		self.estimator_builder = estimator_builder


	def get_scores(self):
		if self.estimator is not None:
			return self.estimator.get_scores()
	
	
	def get_results(self):
		if self.estimator is not None:
			return self.estimator.get_results()


	def _compute(self, estimator=unspecified_argument, **kwargs):
		if estimator is not unspecified_argument:
			self.estimator = estimator
		return super()._compute(**kwargs)


	def _run(self, out=None,):
		out = super()._run(out=out)
		if self.estimator is None:
			self.estimator = self.estimator_builder.build(self.dataset)
		out.update(self.estimator.compute(dataset=self.dataset))
		return out



@fig.Component('task/inference')
class InferenceTaskC(_EncoderTaskC, InferenceTask):
	def __init__(self, A, encoder=unspecified_argument,
	             estimator=unspecified_argument, estimator_builder=unspecified_argument,
	             **kwargs):
		if encoder is unspecified_argument:
			encoder = A.pull('encoder', None, ref=True)
		if estimator_builder is unspecified_argument:
			estimator_builder = A.pull('estimator-builder', None, ref=True)
		super().__init__(A, encoder=encoder, estimator=estimator, estimator_builder=estimator_builder, **kwargs)



# @fig.Component('task/clustering')
class ClusteringTask(InferenceTask): # TODO: specify cluster centers
	pass


class _IterativeTask(Task):
	def __init__(self, batch_size=60, n_samples=15120, pbar=None, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size
		self.n_samples = n_samples
		self.pbar = pbar


	def _get_title(self):
		return None


	def _prep(self, info):
		return info


	def _generate_batch(self, info):
		return info


	def _process_batch(self, info):
		return info


	def _aggregate_results(self, info):
		return info


	def _run(self, out=None):
		if out is None:
			out = util.TensorDict()

		title = self._get_title()

		pbar = self.pbar
		if pbar is not None:
			pbar = pbar(total=self.n_samples)
			if title is not None:
				pbar.set_description(title)
		elif title is not None:
			print(title)

		out = self._prep(out)

		with torch.no_grad():
			j = 0
			while j < self.n_samples:
				N = min(self.batch_size, self.n_samples - j)
				out.num = N

				out = self._generate_batch(out)
				out = self._process_batch(out)

				# out.samples = self._generate_batch(out)
				# out.score = self._judge_batch(out)
				# if scores is not None:
				# 	scores.append(score)

				j += N
				if pbar is not None:
					pbar.update(N)

		if pbar is not None:
			pbar.close()

		return self._aggregate_results(out)


class _IterativeTaskC(TaskC, _IterativeTask):
	def __init__(self, A, batch_size=None, n_samples=None, pbar=unspecified_argument, **kwargs):

		if n_samples is None:
			n_samples = A.pull('n_samples', 15120)

		if batch_size is None:
			batch_size = A.pull('batch_size', 60)

		if pbar is unspecified_argument:
			pbar = A.pull('pbar', None)

		super().__init__(A, n_samples=n_samples, batch_size=batch_size, pbar=pbar, **kwargs)



class GenerationTask(_IterativeTask, _GeneratorTask):
	def __init__(self, criterion, extractor=None, use_dataset_len=False,
	             compute_missing_reference=True, keep_scores=False, **kwargs):
		super().__init__(**kwargs)
		if extractor is None:
			prt.warning('No features extractor for generative task')
		self.extractor = extractor
		self.criterion = get_loss_type(criterion)

		self._use_dataset_len = use_dataset_len
		self._compute_missing_reference = compute_missing_reference
		self._keep_scores = keep_scores
		self._gen_fn = None


	def get_scores(self):
		return ['score']


	def get_results(self):
		return ['stats', 'reference']


	def _compute(self, criterion=unspecified_argument, extractor=unspecified_argument, **kwargs):
		if criterion is not unspecified_argument:
			self.criterion = criterion
		if extractor is not unspecified_argument:
			self.extractor = extractor
		return super()._compute(criterion=criterion, extractor=extractor, **kwargs)


	def _get_title(self):
		ref = f' (reference)' if self.generator is None else ''
		return f'Evaluating Generative Task{ref}'


	def _prep(self, info):
		if self.generator is None:
			self._gen_fn = self.dataset.to_loader(infinite=True, sample_format='observation',
			                                      batch_size=self.batch_size)
			self._gen_fn = self._gen_fn.demand
		else:
			self._gen_fn = self.generator.generate if hasattr(self.generator, 'generate') else self.generator

		if self._use_dataset_len:
			self.n_samples = len(self.dataset)

		info.scores = []

		return super()._prep(info)


	def _generate_batch(self, info):
		info.samples = self._gen_fn(info.num)
		return super()._generate_batch(info)


	def _process_batch(self, info):
		if 'features' not in info:
			info.criteria = self.judge(info.samples)
		info.scores.append(info.criteria)
		return super()._process_batch(info)


	def _aggregate_results(self, info):
		if 'stats' not in info:
			info.stats = self._aggregate_scores(info)
		if self.generator is not None:
			info = self._compare_to_reference(info)
		if 'scores' in info and not self._keep_scores:
			del info.scores
		return super()._aggregate_results(info)


	def judge(self, samples):
		return samples if self.extractor is None else self.extractor.encode(samples)


	def _aggregate_scores(self, info):
		scores = info.scores
		X = torch.cat(scores)
		return X.mean(0), util.cov(X)


	def _reference_props(self):
		props = dict(n_samples=self.n_samples)
		if self.extractor is not None:
			props.update(self.extractor.get_hparams())
		return props


	def _find_reference_scores(self, info):
		props = self._reference_props()
		try:
			return self.dataset.load_stats(props)
		except FileNotFoundError:
			if self._compute_missing_reference:
				_gen = self.generator
				self.generator = None
				ref_stats = self._run().stats
				self.dataset.save_stats(props, ref_stats)
				self.generator = _gen
				return ref_stats


	def _compare_to_reference(self, info):
		if 'reference' not in info:
			info.reference = self._find_reference_scores(info)
		if self.criterion is None:
			info.score = sum(batch.sum(-1) for batch in info.stats) / self.n_samples
		else:
			info.score = self.criterion(info.stats, info.reference)
		return info



@fig.Component('task/generation')
class GenerationTaskC(_IterativeTaskC, _GeneratorTaskC, GenerationTask):
	def __init__(self, A, criterion=unspecified_argument, extractor=unspecified_argument,
	             use_dataset_len=None, compute_missing_reference=None,
	             batch_size=None, n_samples=None, pbar=unspecified_argument, **kwargs):

		if extractor is unspecified_argument:
			extractor = A.pull('extractor', None, ref=True)

		if criterion is unspecified_argument:
			criterion = A.pull('criterion', None, ref=True)

		if use_dataset_len is None:
			use_dataset_len = A.pull('use_dataset_len', False)

		if n_samples is None:
			n_samples = A.pull('n_samples', 50000, silent=use_dataset_len)

		if compute_missing_reference is None:
			compute_missing_reference = A.pull('compute_missing_reference', True)

		if batch_size is None:
			batch_size = A.pull('batch_size', 50)

		if pbar is unspecified_argument:
			pbar = A.pull('pbar', None)

		super().__init__(A, extractor=extractor, criterion=criterion, use_dataset_len=use_dataset_len,
		                 compute_missing_reference=compute_missing_reference, n_samples=n_samples,
		                 batch_size=batch_size, pbar=pbar, **kwargs)



class _ObservationTask(_IterativeTask):
	def __init__(self, sample_format='observation', **kwargs):
		super().__init__(**kwargs)
		self._sample_format = sample_format


	def _prep(self, info):
		self._dataloader = self.dataset.to_loader(infinite=True, sample_format=self._sample_format,
		                                          batch_size=self.batch_size)
		return super()._prep(info)


	def _generate_batch(self, info):
		info.batch = self._dataloader.demand(info.num)
		return super()._generate_batch(info)


	def _aggregate_results(self, info):
		self._dataloader = None
		return super()._aggregate_results(info)



class _ObservationPairTask(_ObservationTask):
	def _split_batch(self, info):
		info.a, info.b = info.batch.split(2)
		info.a_labels, info.b_labels = info.labels.split(2)


	def _generate_batch(self, info):
		info = super()._generate_batch(info)
		self._split_batch(info)
		return info




class MetricTask(_ObservationPairTask, _EncoderTask):
	def __init__(self, metric='mse', criterion='cosine-similarity',
	             keep_distances=False, sample_format=['observations', 'labels'], **kwargs):
		super().__init__(sample_format=sample_format, **kwargs)
		self.metric = get_loss_type(metric)
		self.criterion = get_loss_type(criterion)

		self._keep_distances = keep_distances


	def get_scores(self):
		return ['score']


	def get_results(self):
		return ['distances', 'trues'] if self._keep_distances else []


	def _prep(self, info):
		info.distances = []
		info.trues = []


	def _split_batch(self, info):
		info.observations, info.labels = info.batch
		del info.batch
		info.a, info.b = info.observations.split(2)
		info.a_labels, info.b_labels = info.labels.split(2)


	def _process_batch(self, info):
		info.distances.append(self.metric.distance(info.a, info.b).squeeze())
		info.trues.append(self.dataset.distance(info.a_labels, info.b_labels).squeeze())
		return super()._process_batch(info)


	def _aggregate_results(self, info):
		info.distances = torch.stack(info.distances, 0).unsqueeze(0)
		info.trues = torch.stack(info.trues, 0).unsqueeze(0)

		info.score = self.criterion(info.distances, info.trues)
		if not self._keep_distances:
			del info.distances, info.trues
		return info



@fig.Component('task/metric')
class MetricTaskC(_IterativeTaskC, _EncoderTaskC, MetricTask):
	def __init__(self, A, metric=unspecified_argument, criterion=unspecified_argument, encoder=unspecified_argument,
	             keep_distances=None, **kwargs):

		if metric is unspecified_argument:
			metric = A.pull('metric', None, ref=True)

		if criterion is unspecified_argument:
			criterion = A.pull('criterion', None, ref=True)

		if keep_distances is None:
			keep_distances = A.pull('keep-distances', False)

		super().__init__(A, metric=metric, criterion=criterion, keep_distances=keep_distances, **kwargs)



# @fig.Component('task/creativity')
class CreativityTask(_GeneratorTask):
	pass



# @fig.Component('task/interpolation')
class InterpolationTask(_ObservationPairTask, _EncoderTask, _DecoderTask):
	def __init__(self, interpolator=None, criterion=None, extractor=None,
	             num_steps=None, **kwargs):
		super().__init__(**kwargs)

		self.extractor = extractor
		self.interpolator = interpolator
		self.criterion = criterion

		self._num_steps = num_steps

		raise NotImplementedError