from pathlib import Path
import numpy as np
import torch

from omnibelt import unspecified_argument, get_printer
import omnifig as fig

from omnilearn import util
from omnilearn.data import DataLike
from omnilearn.op import get_save_dir, framework as fm

from .common import EncoderTask, EncoderTaskC, DownstreamTask


@fig.Component('inference-builder')
class InferenceBuilder(util.Builder):
	def __init__(self, A, default_regressor='gbt-regressor', default_classifier='gbt-classifier',
	             use_mechanisms=None, **kwargs):
		A.push('default-regressor', default_regressor, overwrite=False)
		A.push('default-classifier', default_classifier, overwrite=False)
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



@fig.Component('cluster-builder')
class ClusterBuilder(util.Builder):
	def __init__(self, A, default_cluster='kmeans', **kwargs):
		A.push('default-cluster', default_cluster, overwrite=False)
		super().__init__(A, **kwargs)


	def _build(self, config, dataset, n_clusters=None):
		if n_clusters is None:
			try:
				space = dataset.get_target_space()
			except AttributeError:
				n_clusters = dataset.dout
			else:
				assert isinstance(space, util.CategoricalDim)
				n_clusters = space.n

		config.push('n-clusters', n_clusters)
		config.push('clustering._type', '<>default-cluster', overwrite=False, silent=True)
		return config.pull('clustering')



@fig.Component('outlier-builder')
class OutlierBuilder(ClusterBuilder):
	def __init__(self, A, default_cluster=None, **kwargs):
		if default_cluster is None:
			default_cluster = 'isolation-forest'
		super().__init__(A, default_cluster=default_cluster, **kwargs)



class EstimatorTask(DownstreamTask):
	def __init__(self, estimator=None, estimator_builder=None, eval_mode=None, **kwargs):
		if estimator is not None:
			estimator_builder = None
		super().__init__(**kwargs)
		self.estimator = estimator
		self.estimator_builder = estimator_builder

		self._eval_mode = eval_mode


	def get_scores(self):
		scores = super().get_scores()
		if self.estimator is not None:
			scores.extend(self.estimator.get_scores())
		return scores


	def get_results(self):
		results = super().get_results()
		if self.estimator is not None:
			results.extend(self.estimator.get_results())
		return results


	def _compute(self, estimator=unspecified_argument, **kwargs):
		if estimator is not unspecified_argument:
			self.estimator = estimator
		return super()._compute(**kwargs)


	def _run(self, out=None):
		out = super()._run(out=out)

		if self._eval_mode is not None:
			self.dataset.switch_to(self._eval_mode)
		if self.estimator is None:
			self.estimator = self.estimator_builder.build(self.dataset)
		scores, results = self.estimator.compute(dataset=self.dataset, filter_outputs=False)
		return {**scores, **results}
		# out.update(est_out)
		# return out



class InferenceTask(EstimatorTask):
	pass



class ClusterTask(EstimatorTask):
	pass



class AnomalyDetectionTask(EstimatorTask):
	pass



class EstimatorTaskC(EncoderTaskC, EstimatorTask):
	def __init__(self, A, estimator=unspecified_argument, estimator_builder=unspecified_argument,
	             eval_mode=unspecified_argument, **kwargs):
		if estimator_builder is unspecified_argument:
			estimator_builder = A.pull('estimator-builder', None, ref=True)
		if estimator is unspecified_argument:
			estimator = A.pull('estimator', None, ref=True)
		if eval_mode is unspecified_argument:
			eval_mode = A.pull('eval-mode', None)
		super().__init__(A, estimator=estimator, estimator_builder=estimator_builder, eval_mode=eval_mode, **kwargs)



@fig.Component('task/inference')
class InferenceTaskC(EstimatorTaskC, InferenceTask):
	pass



@fig.Component('task/cluster')
class ClusterTaskC(EstimatorTaskC, ClusterTask):
	pass



@fig.Component('task/anomaly')
class AnomalyDetectionTaskC(EstimatorTaskC, AnomalyDetectionTask):
	pass



# @fig.Component('task/clustering')
# class ClusteringTask(EncoderTask): # TODO: specify cluster centers
# 	pass

