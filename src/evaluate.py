from pathlib import Path

from omnibelt import unspecified_argument
import omnifig as fig

import matplotlib.pyplot as plt

import numpy as np

import torch
# from torch import distributions as distrib

from omnilearn import util
from omnilearn.util import distributions as distrib
from omnilearn.op import get_save_dir
# from omnilearn.eval import Metric
# from omnilearn.data import InterventionSamplerBase

# from .responses import sample_full_interventions, response_mat, conditioned_reponses
# from src.tasks.disentanglement_metrics import metric_beta_vae, metric_factor_vae, mig, dci, sap, \
# 	unsupervised_metrics
# from .tasks.disentanglement_metrics import modularity_explicitness, irs, fairness

# from .common import EncoderTask, EncoderTaskC

	
@fig.Script('eval-metrics', 'Compute disentanglement metrics of a trained model')
def eval_metrics(A, run=None, metrics=None, mode=None,
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
		force_run = A.pull('force-run', force_save)
	
	if log_stats is unspecified_argument:
		log_stats = A.pull('log-stats', None)
	if log_stats is not None and not isinstance(log_stats, str):
		log_stats = save_ident
	
	if pbar is unspecified_argument:
		pbar = A.pull('pbar', None)
	
	if run is None:
		run = fig.run('load-run', A)
	
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
	
	# if pbar is not None:
	# 	todo = pbar(todo, total=len(metrics))
	for name, metric in metrics.items():
		# if pbar is not None:
		# 	todo.set_description(name)
		print(name)
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



@fig.Script('eval-multiple-metrics')
def _eval_metrics(A, runs=None, dataset=unspecified_argument, metrics=unspecified_argument):
	
	saveroot = get_save_dir(A)
	
	override = A.pull('override', None, raw=True, silent=True)
	
	if runs is None:
		runs = A.pull('runs', None)
	
	if runs is None:
		run_name = A.pull('run-name')
		runs = [run_name]
	
	if dataset is unspecified_argument:
		dataset = A.pull('dataset', None)
	
	if metrics is unspecified_argument:
		metrics = A.pull('metrics', {})
		if '_list' in metrics:
			metrics.update({metric.get_name():metric for metric in metrics['_list']})
			del metrics['_list']
			
	if runs == 'all':
		runs = list(saveroot.glob('*'))
		
	with A.silenced():
	
		for i, name in enumerate(runs):
			
			run = fig.quick_run('load-run', path=name, saveroot=str(saveroot), override=override)
			
			print(f'Running: {run.get_name()} ({i + 1}/{len(runs)})')
			
			if dataset is None:
				for metric in metrics.values():
					metric.set_dataset(run.get_dataset())
			
			eval_metrics(A, run=run, metrics=metrics)
			#
			# path = root / name
			#
			# if path.is_dir():
			# 	config = fig.get_config(str(path))
			# 	config.push('path', name)
			# 	config.push('saveroot', saveroot)
			# 	if override is not None:
			# 		config.update({'override':override})
			# 	run = config.pull('run')
			#
			# 	print(f'Running: {run.get_name()} ({i+1}/{len(runs)})')
			# 	_eval_run(A, run=run, metrics=metrics)



@fig.Script('eval-fix-hybrid')
def _eval_metrics(A):
	run = fig.run('load-run', A)
	
	model = run.get_model()
	
	# if model._latent is None:
		
	dataset = run.get_dataset()
	dataset.switch_to('train')
	batch = dataset.get_batch(batch_size=128, shuffle=True)
	with torch.no_grad():
		Q = model.encode(batch)
		if isinstance(Q, distrib.Distribution):
			Q = Q.loc
		model._latent = Q
	
	return run.evaluate(config=A)
	



