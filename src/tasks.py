from pathlib import Path

from omnibelt import unspecified_argument
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
from omnilearn.op import get_save_dir, framework as fm#, scikit as sk
from omnilearn.eval import Metric
from omnilearn.data import Supervised, Dataset, Batchable, Deviced, Observation, DatasetWrapper



@DatasetWrapper('encoded')
class Encoded(Observation):
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

		if self.encoder is not None:
			self.set_encoder(encoder)


	def set_encoder(self, encoder):
		self.encoder = encoder
		self.din = getattr(self.encoder, 'latent_dim', getattr(self.encoder, 'dout', None))
		if self._auto_process:
			self._process_observations()


	def _process_observations(self):
		
		loader = DataLoader(self, batch_size=self._batch_size, shuffle=False)

		if self._pbar is not None:
			self._pbar.set_description('Encoding observations')
			loader = self._pbar(loader)
		
		Q = []
		with torch.no_grad():
			for x, *other in loader:
				q = self._encode_observation(x)
				Q.append(q.cpu())
		Q = torch.cat(Q)

		self._observations_encoded = True
		if isinstance(self, Batchable):
			self._replace_observations(Q)
		if isinstance(self, Deviced):
			self.register_buffer('observations', Q)
		else:
			self.observations = Q


	# @DatasetWrapper.condition(Observation)
	def get_observation(self):
		if self._observations_encoded and not isinstance(self, Batchable):
			return self.observations
		return super().get_observations()


	def _encode_observation(self, x, **kwargs):
		q = self.encoder.encode(x.to(self.encoder.get_device()), **kwargs)
		if isinstance(q, distrib.Distribution):
			q = q.bsample() if self._sample_best else q.rsample()
		return q


	def __getitem__(self, item):
		if self._observations_encoded and not isinstance(self, Batchable):
			return self.observations
		x, *other = super().__getitem__(item)
		if not self._observations_encoded:
			x = self._encode_observation(x)
		return x, *other



class Task(Metric, fm.Learnable):
	# def __init__(self, A, model=unspecified_argument, dataset=unspecified_argument, metrics=None, **kwargs):
	#
	# 	if model is unspecified_argument:
	# 		model = A.pull('model', None, ref=True)
	#
	# 	if dataset is unspecified_argument:
	# 		dataset = A.pull('dataset', None, ref=True)
	#
	# 	# if metrics is None:
	# 	# 	metrics = A.pull('metrics', 'all')
	# 	# if metrics == 'all':
	# 	# 	metrics = list(self.KNOWN_METRICS.keys())
	#
	# 	super().__init__(A, **kwargs)

		# self.set_model(model)
		# self.set_dataset(dataset)


	def _compute(self, run):

		A = run.get_config()





		pass



@fig.Component('estimator-builder')
class EstimatorBuilder(util.Builder):
	def __init__(self, A, use_mechanisms=None, **kwargs):
		if use_mechanisms is None:
			use_mechanisms = A.pull('use-mechanisms', False)
		A.push('default-regressor', 'gbt-regressor', overwrite=False)
		A.push('default-classifier', 'gbt-classifier', overwrite=False)
		super().__init__(A, **kwargs)
		self._use_mechanisms = use_mechanisms


	def _build(self, config, space):
		if isinstance(space, Dataset):
			space = space.get_mechanism_space() if self._use_mechanisms else space.get_label_space()

		estimator = None

		if isinstance(space, util.JointSpace):
			config.push('joint._type', 'joint-estimator', overwrite=False, silent=True)
			estimator = config.pull('joint')
			estimator.include_estimators(*[self._build(config, dim) for dim in space])

		if isinstance(space, util.PeriodicDim):
			config.push('periodic._type', '<>default-regressor', overwrite=False, silent=True)
			config.push('periodic._mod.periodized', 1, overwrite=False, silent=True)
			key = 'periodic'

		if isinstance(space, util.DenseDim):
			config.push('continuous._type', '<>default-regressor', overwrite=False, silent=True)
			key = 'continuous'

		if isinstance(space, util.CategoricalDim):
			config.push('categorical._type', '<>default-classifier', overwrite=False, silent=True)
			key = 'categorical'

		if estimator is None:
			estimator = config.pull(key)
			estimator.register_out_space(space)
		return estimator



@fig.Component('task/inference')
class Inference_Task(Task):
	def __init__(self, A, estimator_builder=unspecified_argument, **kwargs):

		if estimator_builder is unspecified_argument:
			estimator_builder = A.pull('estimator-builder', None)

		super().__init__(A, **kwargs)

		self.estimator_builder = estimator_builder
		self.estimator = None


	def get_scores(self):
		if self.estimator is not None:
			return self.estimator.get_scores()
	
	
	def get_results(self):
		if self.estimator is not None:
			return self.estimator.get_results()
	
	
	def _compute(self, info=None, dataset=None):
		if dataset is None:
			dataset = info.get_dataset()
		if self.estimator is None:
			self.estimator = self.estimator_builder.resolve()
		return self.estimator.compute(info, dataset=dataset)
		



