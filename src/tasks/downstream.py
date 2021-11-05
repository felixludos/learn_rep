from pathlib import Path
import numpy as np
import torch

from omnibelt import unspecified_argument, get_printer
import omnifig as fig

from omnilearn import util
from omnilearn.data import DataLike
from omnilearn.op import get_save_dir, framework as fm

from .common import EncoderTask, EncoderTaskC


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



class DownstreamTask(EncoderTask):
	def _run(self, out=None):
		out = super()._run(out=out)
		if self.encoder is not None:
			self.dataset.register_wrapper('encoded', encoder=self.encoder)
		return out



class InferenceTask(DownstreamTask):
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


	def _run(self, out=None, ):
		out = super()._run(out=out)
		if self.estimator is None:
			self.estimator = self.estimator_builder.build(self.dataset)
		out.update(self.estimator.compute(dataset=self.dataset))
		return out



@fig.Component('task/inference')
class InferenceTaskC(EncoderTaskC, InferenceTask):
	def __init__(self, A, encoder=unspecified_argument,
	             estimator=unspecified_argument, estimator_builder=unspecified_argument,
	             **kwargs):
		if encoder is unspecified_argument:
			encoder = A.pull('encoder', None, ref=True)
		if estimator_builder is unspecified_argument:
			estimator_builder = A.pull('estimator-builder', None, ref=True)
		super().__init__(A, encoder=encoder, estimator=estimator, estimator_builder=estimator_builder, **kwargs)




# @fig.Component('task/clustering')
# class ClusteringTask(EncoderTask): # TODO: specify cluster centers
# 	pass

