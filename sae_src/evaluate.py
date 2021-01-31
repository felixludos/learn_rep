import sys, os
from pathlib import Path
from tqdm import tqdm

from omnibelt import unspecified_argument
import omnifig as fig

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import distributions as distrib

from foundation import util
from foundation.eval import Evaluator

from .responses import sample_full_interventions, response_mat, factor_reponses
from .metrics import metric_beta_vae, metric_factor_vae, mig, dci, irs, sap, \
	modularity_explicitness, unsupervised_metrics, fairness


class Disentanglement_Evaluator(Evaluator, util.Seed, util.Switchable, util.Deviced):
	# TODO: turn into an alert and stats client
	
	# KNOWN_METRICS = {
	# 	'beta-vae': eval_beta_vae,
	# 	'factor-vae': eval_factor_vae,
	# 	'mig': eval_mig,
	# 	'dci': eval_dci,
	# 	'irs': eval_irs,
	# 	'sap': eval_sap,
	# 	'modularity-explicitness': eval_modularity_explicitness,
	# 	'unsupervised': eval_unsupervised,
	# }
	
	def __init__(self, A, model=unspecified_argument, dataset=unspecified_argument, metrics=None, **kwargs):
		
		if model is unspecified_argument:
			model = A.pull('model', None, ref=True)
		
		if dataset is unspecified_argument:
			dataset = A.pull('dataset', None, ref=True)
		
		# if metrics is None:
		# 	metrics = A.pull('metrics', 'all')
		# if metrics == 'all':
		# 	metrics = list(self.KNOWN_METRICS.keys())
		
		super().__init__(A, **kwargs)
		
		self.set_model(model)
		self.set_dataset(dataset)
	
	def get_name(self):
		return self.__class__.__name__
	
	# self.metrics = metrics
	
	def compute(self, info=None):
		assert self.model is not None
		# assert self.dataset is not None
		self.model.switch_to('eval')
		util.set_seed(self.seed)
		return super().compute(info=info)
	
	def set_model(self, model=None):
		self.model = model
	
	def set_dataset(self, dataset=None):
		self.dataset = dataset
	
	def _representation_function(self, images):
		with torch.no_grad():
			output = self.model.encode(images.to(self.get_device()))
		if isinstance(output, distrib.Normal):
			output = output.loc
		return output.detach().cpu().numpy()

@fig.Component('metric/unsupervised')
class UnsupervisedMetrics(Disentanglement_Evaluator):
	def __init__(self, A, num_train=None, batch_size=None, **kwargs):
		
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		
		if batch_size is None:
			batch_size = A.pull('batch_size', 64)
		
		super().__init__(A, **kwargs)
		
		self.num_train = num_train
		self.batch_size = batch_size
		
	def _compute(self, info=None):
		return unsupervised_metrics.unsupervised_metrics(self.dataset, self._representation_function,
		                                                 np.random, self.num_train, self.batch_size)
		
	def get_scores(self):
		return ['gaussian_total_correlation', 'gaussian_wasserstein_correlation',
		        'gaussian_wasserstein_correlation_norm']
	
	def get_results(self):
		return ['covariance_matrix']

@fig.Component('metric/modularity-explicitness')
class ModularityExplicitness(Disentanglement_Evaluator):
	def __init__(self, A, num_train=None, num_test=None, batch_size=None, **kwargs):
		
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		
		if num_test is None:
			num_test = A.pull('num_test', 5000)
		
		if batch_size is None:
			batch_size = A.pull('batch_size', 64)
		
		super().__init__(A, **kwargs)
		
		self.num_train = num_train
		self.num_test = num_test
		self.batch_size = batch_size
	
	def _compute(self, info=None):
		return modularity_explicitness.compute_modularity_explicitness(self.dataset, self._representation_function,
		                                                 np.random, self.num_train, self.num_test, self.batch_size)
	
	def get_scores(self):
		return ['modularity_score', 'explicitness_score_train', 'explicitness_score_test']

@fig.Component('metric/sap')
class SAP(Disentanglement_Evaluator):
	def __init__(self, A, num_train=None, num_test=None, batch_size=None, continuous_factors=None, **kwargs):
		
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		
		if num_test is None:
			num_test = A.pull('num_test', 5000)
		
		if batch_size is None:
			batch_size = A.pull('batch_size', 64)
		
		if continuous_factors:
			continuous_factors = A.pull('continuous_factors', False)
		
		super().__init__(A, **kwargs)
		
		self.num_train = num_train
		self.num_test = num_test
		self.batch_size = batch_size
		self.continuous_factors = continuous_factors
	
	def _compute(self, info=None):
		return sap.compute_sap(self.dataset, self._representation_function, np.random,
		                       self.num_train, self.num_test, self.continuous_factors, self.batch_size)
	
	def get_scores(self):
		return ['SAP_score']
	
	def get_results(self):
		return ['SAP_matrix']

@fig.Component('metric/irs')
class IRS(Disentanglement_Evaluator):
	def __init__(self, A, num_train=None, batch_size=None, diff_quantile=None, **kwargs):
		
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		
		if batch_size is None:
			batch_size = A.pull('batch_size', 64)
		
		if diff_quantile is None:
			diff_quantile = A.pull('diff_quantile', 0.99)
		
		super().__init__(A, **kwargs)
		
		self.num_train = num_train
		self.batch_size = batch_size
		self.diff_quantile = diff_quantile
	
	def _compute(self, info=None):
		return irs.compute_irs(self.dataset, self._representation_function, np.random,
		                       self.num_train, self.batch_size, self.diff_quantile)
	
	def get_scores(self):
		return ['avg_score', 'num_active_dims', ]
	
	def get_results(self):
		return ['IRS_matrix', 'disentanglement_scores', 'parents', 'max_deviations']

@fig.Component('metric/dci')
class DCI(Disentanglement_Evaluator):
	def __init__(self, A, num_train=None, num_test=None, batch_size=None, **kwargs):
		
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		
		if num_test is None:
			num_test = A.pull('num_test', 5000)
		
		if batch_size is None:
			batch_size = A.pull('batch_size', 64)
		
		super().__init__(A, **kwargs)
		
		self.num_train = num_train
		self.num_test = num_test
		self.batch_size = batch_size
	
	def _compute(self, info=None):
		return dci.compute_dci(self.dataset, self._representation_function, np.random,
		                       self.num_train, self.num_test, self.batch_size)
	
	def get_scores(self):
		return ['informativeness_train', 'informativeness_test', 'disentanglement', 'completeness']
	
	def get_results(self):
		return ['importance_matrix']

@fig.Component('metric/mig')
class MIG(Disentanglement_Evaluator):
	def __init__(self, A, num_train=None, batch_size=None, **kwargs):
		
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		
		if batch_size is None:
			batch_size = A.pull('batch_size', 64)
		
		super().__init__(A, **kwargs)
		
		self.num_train = num_train
		self.batch_size = batch_size
	
	def _compute(self, info=None):
		return mig.compute_mig(self.dataset, self._representation_function, np.random,
		                       self.num_train, self.batch_size)
	
	def get_scores(self):
		return ['discrete_mig']
	
	def get_results(self):
		return ['entropy']

@fig.Component('metric/factor-vae')
class FactorVAE(Disentanglement_Evaluator):
	def __init__(self, A, num_train=None, num_test=None, num_variance_estimate=None, batch_size=None, **kwargs):
		
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		
		if num_test is None:
			num_test = A.pull('num_test', 5000)

		if num_variance_estimate is None:
			num_variance_estimate = A.pull('num_variance_estimate', 10000)
			
		if batch_size is None:
			batch_size = A.pull('batch_size', 64)
		
		super().__init__(A, **kwargs)
		
		self.num_train = num_train
		self.num_test = num_test
		self.num_variance_estimate = num_variance_estimate
		self.batch_size = batch_size
	
	def _compute(self, info=None):
		return metric_factor_vae.compute_factor_vae(self.dataset, self._representation_function, np.random,
		                       self.batch_size, self.num_train, self.num_test, self.num_variance_estimate)
	
	def get_scores(self):
		return ['train_accuracy', 'eval_accuracy', 'num_active_dims']

@fig.Component('metric/beta-vae')
class BetaVAE(Disentanglement_Evaluator):
	def __init__(self, A, num_train=None, num_test=None, batch_size=None, **kwargs):
		
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		
		if num_test is None:
			num_test = A.pull('num_test', 5000)
		
		if batch_size is None:
			batch_size = A.pull('batch_size', 64)
		
		super().__init__(A, **kwargs)
		
		self.num_train = num_train
		self.num_test = num_test
		self.batch_size = batch_size
	
	def _compute(self, info=None):
		return metric_beta_vae.compute_beta_vae_sklearn(self.dataset, self._representation_function, np.random,
		                       self.num_train, self.num_test, self.batch_size)
	
	def get_scores(self):
		return ['train_accuracy', 'eval_accuracy']

@fig.Component('metric/fairness')
class Fairness(Disentanglement_Evaluator):
	def __init__(self, A, num_train=None, num_test_points_per_class=None, batch_size=None, **kwargs):
		
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		
		if num_test_points_per_class is None:
			num_test_points_per_class = A.pull('num_test_points_per_class', 100)
		
		if batch_size is None:
			batch_size = A.pull('batch_size', 64)
		
		super().__init__(A, **kwargs)
		
		self.num_train = num_train
		self.num_test_points_per_class = num_test_points_per_class
		self.batch_size = batch_size
	
	def _compute(self, info=None):
		return fairness.compute_fairness(self.dataset, self._representation_function, np.random,
		                       self.num_train, self.num_test_points_per_class, self.batch_size)
	
	def get_scores(self):
		return ['informativeness_train', 'informativeness_test', 'disentanglement', 'completeness']
	
	def get_results(self):
		return ['importance_matrix']


@fig.Component('metric/responses')
class LatentResponses(Disentanglement_Evaluator):
	def __init__(self, A, **kwargs):
		
		num_groups = A.pull('num_groups', 50)
		num_q = A.pull('num_latent', 10000)
		num_resp = A.pull('num_response', 100)
		batch_size = A.pull('batch_size', 64)
		
		dist_type = A.pull('dist-type', 'rms')
		force_different = A.pull('force-different', True)

		normalize = A.pull('normalize', True)
		include_q = A.pull('include-q', True)
		
		
		figure_dir = A.pull('figure_dir', None)
		if figure_dir is not None:
			figure_dir = Path(figure_dir)
			if not figure_dir.is_dir():
				figure_dir.mkdir()
		
		pbar = A.pull('pbar', None)
		
		super().__init__(A, **kwargs)

		self.figure_dir = figure_dir

		self.num_groups = num_groups
		self.num_q = num_q
		self.num_resp = num_resp
		self.batch_size = batch_size
		self.pbar = pbar
		
		self.dist_type = dist_type
		self.force_different = force_different
		self.normalize = normalize
		
		self.include_q = include_q
		
		self.interventions = None
		
	def get_results(self):
		return ['response_mat', 'covariance', 'factor_responses', 'factor_responses_q']
		
	def get_scores(self):
		return ['disentanglement']
		
	def _compute(self, info):
		
		run_name = info.get_name()
		
		model = self.model
		
		fullQ = []
		total = self.num_q
		bs = self.batch_size
		loader = self.dataset.get_loader(infinite=True, shuffle=True, seed=0, batch_size=bs)
		loader = iter(loader)
		pbar = tqdm(total=total)
		while len(fullQ) < total // bs:
			batch = next(loader)
			x = model._process_batch(batch).original
			with torch.no_grad():
				q = model.encode(x)
				if isinstance(q, distrib.Distribution):
					q = q.loc
				fullQ.append(q)
			pbar.update(bs)
		del loader
		fullQ = torch.cat(fullQ)

		scales = fullQ.std(0) if self.normalize else None
		
		C = np.cov(fullQ.cpu().t().numpy())
		if self.figure_dir is not None:
			util.plot_mat(C, val_fmt=2)
			plt.tight_layout()
			util.save_figure(f'{run_name}_cov', root=self.figure_dir)
		
		R = response_mat(fullQ[:self.num_resp], model.encode, model.decode, scales=scales,
		                 dist_type='rms', force_different=True)

		if self.figure_dir is not None:
			util.plot_mat(R, val_fmt=1)  # responses
			plt.ylabel('Intervention')
			plt.xlabel('Response')
			plt.tight_layout()
			util.save_figure(f'{run_name}_responses', root=self.figure_dir)
			
			util.plot_mat((R @ R.t()), val_fmt=1)  # interactions?
			plt.tight_layout()
			util.save_figure(f'{run_name}_interactions', root=self.figure_dir)
		

		if self.interventions is None:
			self.interventions = sample_full_interventions(self.dataset, num_groups=self.num_groups, pbar=self.pbar)

		
		out = factor_reponses(model.encode, model.decode, self.interventions, pbar=self.pbar,
		                            include_q=self.include_q,
		                            resp_kwargs=dict(scales=scales, force_different=self.force_different))

		if self.include_q:
			mats, lts = out
		else:
			mats, lts = out, None
		
		M = mats.min(1)[0].max(-1)[0]
		if self.figure_dir is not None:
			factors = self.dataset.get_factor_order()
			util.plot_mat(M, val_fmt=1)
			plt.yticks(range(len(factors)), factors)
			plt.xlabel('Latent dimension')
			plt.tight_layout()
			util.save_figure(f'{run_name}_factor-responses', root=self.figure_dir)
		
		disentanglement = M.max(0)[0].sum() / M.sum()
		
		return {'disentanglement': disentanglement}, \
		       {'response_mat':R, 'covariance':C, 'factor_responses':mats, 'factor_responses_q':lts,}
	
@fig.Script('eval-metrics')
def _eval_run(A, run=None, metrics=None, mode=None,
              force_run=None, force_save=None, log_stats=unspecified_argument,
              save_ident=unspecified_argument, pbar=unspecified_argument):
	
	if save_ident is unspecified_argument:
		save_ident = A.pull('save-ident', None)
	
	if mode is None:
		mode = 'eval' if save_ident is None else save_ident
		mode = A.pull('mode', mode)
	
	if force_save is None:
		force_save = A.pull('force-save', False)
	if force_run is None:
		force_run = A.pull('force-run', False)
	
	if log_stats is unspecified_argument:
		log_stats = A.pull('log-stats', None)
	if log_stats is not None and not isinstance(log_stats, str):
		log_stats = save_ident
	
	if pbar is unspecified_argument:
		pbar = A.pull('pbar', None)
	
	if run is None:
		run = A.pull('run')
	
	if not force_run and (save_ident is None or run.has_results(save_ident)):
		print(f'  skipping: {run.get_name()}')
		return
	
	if metrics is None:
		metrics = A.pull('metrics')
		if '_list' in metrics:
			metrics.update({metric.get_name():metric for metric in metrics['_list']})
			del metrics['_list']
	
	model = run.get_model()
	model.switch_to(mode)
	
	scores = {}
	results = {}
	
	todo = metrics.items()
	if pbar is not None:
		todo = pbar(todo, total=len(metrics))
	for name, metric in todo:
		if pbar is not None:
			todo.set_description(name)
		metric.set_model(model)
		
		score, result = metric.compute(run)
		
		scores[name] = score
		results[name] = result
	
	if save_ident is not None and (force_save or not run.has_results(save_ident)):
		run.update_results(save_ident, {'scores': scores, 'results': results})
	
	if log_stats:
		records = run.get_records()
		records.switch_to(mode)
		records.set_fmt(f'{mode}/' + '{}')
		records.set_step(run.get_clock().get_time())
		
		for metric, score in scores.items():
			for name, val in score.items():
				if val is not None:
					records.log('scalar', f'{metric}-{name}', val)
		
	
	return scores, results


@fig.Script('eval-multiple-runs')
def _eval_metrics(A, runs=None, dataset=unspecified_argument, metrics=unspecified_argument):
	
	saveroot = A.pull('saveroot', os.environ.get('FOUNDATION_SAVE_DIR', '.'))
	root = Path(saveroot)
	
	override = A.pull('override', None, raw=True, silent=True)
	
	if runs is None:
		runs = A.pull('runs')
	
	if dataset is unspecified_argument:
		dataset = A.pull('dataset', None)
	
	if metrics is unspecified_argument:
		metrics = A.pull('metrics', {})
		if '_list' in metrics:
			metrics.update({metric.get_name():metric for metric in metrics['_list']})
			del metrics['_list']
			
	if runs == 'all':
		runs = list(root.glob('*'))
		
	with A.silenced():
	
		for i, name in enumerate(runs):
			
			path = root / name
			
			if path.is_dir():
				config = fig.get_config(str(path))
				config.push('path', name)
				config.push('saveroot', saveroot)
				if override is not None:
					config.update({'override':override})
				run = config.pull('run')
				
				print(f'Running: {run.get_name()} ({i+1}/{len(runs)})')
				_eval_run(A, run=run, metrics=metrics)


@fig.Script('eval-fid')
def _eval_fid(A, run=None, fid=None, dataset=None):
	if run is None:
		run = A.pull('run')
	
	if fid is None:
		fid = A.pull('fid', ref=True)
	
	ident = A.pull('ident', 'eval')
	overwrite = A.pull('overwrite', False)
	
	model = None
	results = run.get_results(ident)
	out = None if overwrite or results is None else util.TensorDict(results)
	
	if 'rec_fid' not in results:
		
		model = run.get_model()
		if dataset is None:
			dataset = run.get_dataset()
		else:
			run.dataset = dataset
	
		out = model.evaluate(run, A, out=out)
		
	
		run.update_results(ident, out)
	

@fig.Script('eval-multiple-fids')
def _eval_all_fid(A, runs=None, dataset=unspecified_argument, fid=None):

	saveroot = A.pull('saveroot', os.environ.get('FOUNDATION_SAVE_DIR', '.'))
	root = Path(saveroot)

	override = A.pull('override', None, raw=True, silent=True)
	
	if runs is None:
		runs = A.pull('runs')

	if dataset is unspecified_argument:
		dataset = A.pull('dataset', None)

	if fid is None:
		fid = A.pull('fid', ref=True)
	
	with A.silenced():
		
		for i, name in enumerate(runs):
			
			path = root / name
			
			if path.is_dir():
				config = fig.get_config(str(path))
				config.push('path', name)
				config.push('saveroot', saveroot)
				if override is not None:
					config.update({'override': override})
				run = config.pull('run')
				
				print(f'Running: {run.get_name()} ({i + 1}/{len(runs)})')
				_eval_fid(A, run=run, fid=fid, dataset=dataset)
	pass

