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


	def _run(self, **kwargs):
		raise NotImplementedError



class TaskC(fig.Configurable, Task):
	def __init__(self, A, dataset=unspecified_argument, **kwargs):
		if dataset is unspecified_argument:
			dataset = A.pull('dataset', None, ref=True)
		super().__init__(A, dataset=dataset, **kwargs)



class _EncoderTask(Task):
	def __init__(self, encoder=None, **kwargs):
		super().__init__(**kwargs)
		self.encoder = encoder


	def _compute(self, encoder=unspecified_argument, **kwargs):
		if encoder is not unspecified_argument:
			self.encoder = encoder
		return super()._compute(**kwargs)



class _GeneratorTask(Task):
	def __init__(self, generator=None, **kwargs):
		super().__init__(**kwargs)
		self.generator = generator


	def _compute(self, generator=unspecified_argument, **kwargs):
		if generator is not unspecified_argument:
			self.generator = generator
		return super()._compute(**kwargs)



class InferenceTask(_EncoderTask):
	def __init__(self, estimator=None, **kwargs):
		super().__init__(**kwargs)
		self.estimator = estimator


	def get_scores(self):
		if self.estimator is not None:
			return self.estimator.get_scores()
	
	
	def get_results(self):
		if self.estimator is not None:
			return self.estimator.get_results()


	def _compute(self, estimator=None, **kwargs): # self.estimator_builder.build(dataset)
		if estimator is None:
			estimator = self.estimator
		return super()._compute(estimator=estimator, **kwargs)


	def _run(self, dataset, encoder, estimator, **kwargs):
		if encoder is not None:
			dataset.register_wrapper('encoded', encoder=encoder)
		if self.estimator is None:
			self.estimator = self.estimator_builder.build(dataset)
		return estimator.compute(dataset=dataset)



@fig.Component('task/inference')
class InferenceTaskC(TaskC, InferenceTask):
	def __init__(self, A, encoder=unspecified_argument,
	             estimator=unspecified_argument, estimator_builder=unspecified_argument,
	             **kwargs):
		if encoder is unspecified_argument:
			encoder = A.pull('encoder', None, ref=True)
		if estimator is unspecified_argument:
			estimator = A.pull('estimator', None, ref=True) \
				if estimator_builder in {unspecified_argument, None} \
				else None
		if estimator is not None:
			estimator_builder = None
		elif estimator_builder is unspecified_argument:
			estimator_builder = A.pull('estimator-builder', None, ref=True)
		super().__init__(A, encoder=encoder, estimator=estimator, **kwargs)
		self.estimator_builder = estimator_builder


	def _compute(self, dataset=None, estimator=None, estimator_builder=None, **kwargs):
		if dataset is None:
			dataset = self.dataset
		if estimator is None:
			estimator = self.estimator
		if estimator_builder is None:
			estimator_builder = self.estimator_builder
		if estimator is None:
			estimator = estimator_builder.build(dataset)
		return super()._compute(dataset=dataset, estimator=estimator, **kwargs)



@fig.Component('task/clustering')
class ClusteringTask(InferenceTask): # TODO: specify cluster centers
	pass



@fig.Component('task/generation')
class GenerationTask(_GeneratorTask):
	def __init__(self, criterion, extractor=None,
	             use_dataset_len=False, compute_missing_reference=True,
	             batch_size=50, n_samples=50000, pbar=None, **kwargs):
		super().__init__(**kwargs)
		if extractor is None:
			prt.warning('No features extractor for generative task')
		self.extractor = extractor
		self.criterion = get_loss_type(criterion)

		self._use_dataset_len = use_dataset_len
		self._compute_missing_reference = compute_missing_reference
		self._accumulation = None
		self.batch_size = batch_size
		self.n_samples = n_samples
		self.pbar = pbar
		self._gen_fn = None


	def get_scores(self):
		return ['score']


	def _compute(self, criterion=None, extractor=None, **kwargs):
		if criterion is None:
			criterion = self.criterion
		if extractor is None:
			extractor = self.extractor
		return super()._compute(criterion=criterion, extractor=extractor, **kwargs)


	def _run(self, dataset, generator, criterion, extractor=None, **kwargs):



		# if encoder is not None:
		# 	dataset.register_wrapper('encoded', encoder=encoder)
		# return estimator.compute(dataset=dataset)
		pass


	def get_results(self):
		return ['stats', 'reference']


	def _get_title(self):
		return 'Evaluating Generative Task'


	def generate(self, N, **kwargs):
		return self._gen_fn(N, **kwargs)


	def judge(self, samples):
		if self.extractor is not None:
			samples = self.extractor.encode(samples)
		return samples


	def _prep_collection(self):
		self._gen_fn = self.model.generate if hasattr(self.model, 'generate') else self.model

		if self.model is None:
			self._gen_fn = self.dataset.to_loader(infinite=True, sample_format='observation',
			                                      batch_size=self.batch_size)
			self._gen_fn = self._gen_fn.demand

		if self._use_dataset_len:
			self.n_samples = len(self.dataset)

		return []


	def aggregate(self, scores, out=None):
		if out is None:
			out = util.TensorDict()
		if 'stats' not in out:
			out.stats = self._aggregate_scores(scores)
		if self.model is not None:
			out = self._compare_to_reference(out)
		return out


	def _collect(self):
		title = self._get_title()

		pbar = self.pbar
		if pbar is not None:
			pbar = pbar(total=self.n_samples)
			pbar.set_description(title)
		elif title is not None:
			print(title)

		scores = self._prep_collection()

		with torch.no_grad():
			j = 0
			while j < self.n_samples:
				N = min(self.batch_size, self.n_samples - j)

				samples = self.generate(N)
				score = self.judge(samples)
				if scores is not None:
					scores.append(score)

				j += N
				if pbar is not None:
					pbar.update(N)

		if pbar is not None:
			pbar.close()

		return self.aggregate(scores)


	def set_dataset(self, dataset=None):
		self.dataset = dataset


	def _aggregate_scores(self, scores):
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
				_model = self.model
				self.model = None
				ref_stats = self._collect().stats
				self.dataset.save_stats(props, ref_stats)
				self.model = _model
				return ref_stats


	def _compare_to_reference(self, info):
		if 'reference' not in info:
			info.reference = self._find_reference_scores(info)
		if self.criterion is None:
			info.score = sum(batch.sum(-1) for batch in info.stats) / self.n_samples
		else:
			info.score = info.stats[0]
		return info



class GenerationTaskC(TaskC, GenerationTask):
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

		super().__init__(A, **kwargs)

		if extractor is None:
			prt.warning('No features extractor for generative task')
		self.extractor = extractor
		self.criterion = get_loss_type(criterion)

		self._use_dataset_len = use_dataset_len
		self._compute_missing_reference = compute_missing_reference
		self._accumulation = None
		self.batch_size = batch_size
		self.n_samples = n_samples
		self.pbar = pbar
		self._gen_fn = None




@fig.Component('task/metric')
class MetricTask(Task):
	def __init__(self, A, model=unspecified_argument, metric=unspecified_argument,
	             dataset=unspecified_argument, criterion=unspecified_argument,
	             batch_size=None, n_samples=None, pbar=unspecified_argument, **kwargs):

		if model is unspecified_argument:
			model = A.pull('model', None, ref=True)

		if dataset is unspecified_argument:
			dataset = A.pull('dataset', None, ref=True)

		if metric is unspecified_argument:
			metric = A.pull('metric', None, ref=True)

		if criterion is unspecified_argument:
			criterion = A.pull('criterion', None, ref=True)

		if n_samples is None:
			n_samples = A.pull('n_samples', 50000)

		if batch_size is None:
			batch_size = A.pull('batch_size', 50)

		if pbar is unspecified_argument:
			pbar = A.pull('pbar', None)

		super().__init__(A, **kwargs)

		self.set_model(model)
		self.set_dataset(dataset)

		if metric is None:
			prt.warning('No metric provided')
		self.metric = metric
		self.criterion = get_loss_type(criterion)

		self.batch_size = batch_size
		self.n_samples = n_samples
		self.pbar = pbar
		self._gen_fn = None


	def learned_distance(self, a, b):
		return (self.model if self.metric is None else self.metric).distance(a, b)


	def true_distance(self, a, b):
		return self.dataset.distance(a, b)


	def _compute(self, info=None, model=None, dataset=None):
		pass



@fig.Component('task/creativity')
class CreativityTask(Task):
	pass



@fig.Component('task/interpolation')
class InterpolationTask(Task):
	def __init__(self, A, **kwargs):

		super().__init__(A, **kwargs)


	def _prep_collection(self):
		assert self.model is not None
		return super()._prep_collection()


	def sample_points(self, N, **kwargs):

		pass

	def interpolate(self):
		pass




