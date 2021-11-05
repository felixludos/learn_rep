from pathlib import Path
import numpy as np
import torch

from omnibelt import unspecified_argument, get_printer
import omnifig as fig

from omnilearn import util
from omnilearn.util import distributions as distrib
from omnilearn.data import DataLike
from omnilearn.op import get_save_dir, framework as fm


from .disentanglement_metrics import metric_beta_vae, metric_factor_vae, mig, dci, sap, \
	unsupervised_metrics, modularity_explicitness, irs, fairness

from .common import EncoderTask, EncoderTaskC, ObservationTask, IterativeTaskC, DecoderTask, DecoderTaskC



class DisentanglementTask(EncoderTask):
	def __init__(self, batch_size=64, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size


	def get_name(self):
		return self.__class__.__name__


	def _representation_function(self, images):
		with torch.no_grad():
			output = self.model.encode(images.to(self.get_device()))
		if isinstance(output, distrib.Distribution):
			output = output.loc
		return output.detach().cpu().numpy()



class DisentanglementTaskC(EncoderTaskC):
	def __init__(self, A, batch_size=None, **kwargs):
		if batch_size is None:
			batch_size = A.pull('batch_size', 64)
		super().__init__(A, batch_size=batch_size, **kwargs)



class UnsupervisedMetrics(DisentanglementTask):
	def __init__(self, num_train=10000, **kwargs):
		super().__init__(**kwargs)
		self.num_train = num_train


	def _compute(self, **kwargs):
		return unsupervised_metrics.unsupervised_metrics(self.dataset, self._representation_function,
		                                                 np.random, self.num_train, self.batch_size)


	def get_scores(self):
		return ['gaussian_total_correlation', 'gaussian_wasserstein_correlation',
		        'gaussian_wasserstein_correlation_norm']


	def get_results(self):
		return ['covariance_matrix']



@fig.Component('task/disentanglement/unsupervised')
class UnsupervisedMetricsC(DisentanglementTaskC, UnsupervisedMetrics):
	def __init__(self, A, num_train=None, **kwargs):
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		super().__init__(A, num_train=num_train, **kwargs)



class ModularityExplicitness(DisentanglementTask):
	def __init__(self, num_train=10000, num_test=5000, **kwargs):
		super().__init__(**kwargs)
		self.num_train = num_train
		self.num_test = num_test


	def _compute(self, **kwargs):
		return modularity_explicitness.compute_modularity_explicitness(self.dataset, self._representation_function,
		                                                               np.random, self.num_train, self.num_test,
		                                                               self.batch_size)


	def get_scores(self):
		return ['modularity_score', 'explicitness_score_train', 'explicitness_score_test']



@fig.Component('task/disentanglement/mod-exp')
class ModularityExplicitnessC(DisentanglementTaskC, ModularityExplicitness):
	def __init__(self, A, num_train=None, num_test=None, **kwargs):
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		if num_test is None:
			num_test = A.pull('num_test', 5000)
		super().__init__(A, num_train=num_train, num_test=num_test, **kwargs)



class SAP(DisentanglementTask):
	def __init__(self, num_train=10000, num_test=5000, continuous_factors=False, **kwargs):
		super().__init__(**kwargs)

		self.num_train = num_train
		self.num_test = num_test
		self.continuous_factors = continuous_factors


	def _compute(self, info=None):
		return sap.compute_sap(self.dataset, self._representation_function, np.random,
		                       self.num_train, self.num_test, self.continuous_factors, self.batch_size)


	def get_scores(self):
		return ['SAP_score']


	def get_results(self):
		return ['SAP_matrix']



@fig.Component('task/disentanglement/sap')
class SAPc(DisentanglementTaskC, SAP):
	def __init__(self, A, num_train=None, num_test=None, continuous_factors=None, **kwargs):
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		if num_test is None:
			num_test = A.pull('num_test', 5000)
		if continuous_factors:
			continuous_factors = A.pull('continuous_factors', False)
		super().__init__(A, num_train=num_train, num_test=num_test, continuous_factors=continuous_factors, **kwargs)



class IRS(DisentanglementTask):
	def __init__(self, num_train=10000, diff_quantile=0.99, **kwargs):
		super().__init__(**kwargs)
		self.num_train = num_train
		self.diff_quantile = diff_quantile


	def _compute(self, **kwargs):
		return irs.compute_irs(self.dataset, self._representation_function, np.random,
		                       self.num_train, self.batch_size, self.diff_quantile)


	def get_scores(self):
		return ['avg_score', 'num_active_dims', ]


	def get_results(self):
		return ['IRS_matrix', 'disentanglement_scores', 'parents', 'max_deviations']



@fig.Component('task/disentanglement/irs')
class IRSc(DisentanglementTaskC, IRS):
	def __init__(self, A, num_train=None, diff_quantile=None, **kwargs):
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		if diff_quantile is None:
			diff_quantile = A.pull('diff_quantile', 0.99)
		super().__init__(A, num_train=num_train, diff_quantile=diff_quantile, **kwargs)



class DCI(DisentanglementTask):
	def __init__(self, num_train=10000, num_test=5000, **kwargs):
		super().__init__(**kwargs)
		self.num_train = num_train
		self.num_test = num_test


	def _compute(self, **kwargs):
		return dci.compute_dci(self.dataset, self._representation_function, np.random,
		                       self.num_train, self.num_test, self.batch_size)


	def get_scores(self):
		return ['informativeness_train', 'informativeness_test', 'disentanglement', 'completeness']


	def get_results(self):
		return ['importance_matrix']



@fig.Component('task/disentanglement/dci')
class DCIc(DisentanglementTaskC, DCI):
	def __init__(self, A, num_train=None, num_test=None, **kwargs):
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		if num_test is None:
			num_test = A.pull('num_test', 5000)
		super().__init__(A, num_train=num_train, num_test=num_test, **kwargs)



class MIG(DisentanglementTask):
	def __init__(self, num_train=10000, **kwargs):
		super().__init__(**kwargs)
		self.num_train = num_train


	def _compute(self, **kwargs):
		return mig.compute_mig(self.dataset, self._representation_function, np.random,
		                       self.num_train, self.batch_size)


	def get_scores(self):
		return ['discrete_mig']


	def get_results(self):
		return ['entropy']



@fig.Component('task/disentanglement/mig')
class MIGc(DisentanglementTaskC, MIG):
	def __init__(self, A, num_train=None, **kwargs):
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		super().__init__(A, num_train=num_train, **kwargs)



class FactorVAE(DisentanglementTask):
	def __init__(self, num_train=10000, num_test=5000, num_variance_estimate=10000, **kwargs):
		super().__init__(**kwargs)

		self.num_train = num_train
		self.num_test = num_test
		self.num_variance_estimate = num_variance_estimate


	def _compute(self, **kwargs):
		return metric_factor_vae.compute_factor_vae(self.dataset, self._representation_function, np.random,
		                                            self.batch_size, self.num_train, self.num_test,
		                                            self.num_variance_estimate)


	def get_scores(self):
		return ['train_accuracy', 'eval_accuracy', 'num_active_dims']



@fig.Component('task/disentanglement/factor-vae')
class FactorVAEc(DisentanglementTaskC, FactorVAE):
	def __init__(self, A, num_train=None, num_test=None, num_variance_estimate=None, **kwargs):
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		if num_test is None:
			num_test = A.pull('num_test', 5000)
		if num_variance_estimate is None:
			num_variance_estimate = A.pull('num_variance_estimate', 10000)
		super().__init__(A, num_train=num_train, num_test=num_test, num_variance_estimate=num_variance_estimate,
		                 **kwargs)



class BetaVAE(DisentanglementTask):
	def __init__(self, num_train=10000, num_test=5000, **kwargs):
		super().__init__(**kwargs)

		self.num_train = num_train
		self.num_test = num_test


	def _compute(self, **kwargs):
		return metric_beta_vae.compute_beta_vae_sklearn(self.dataset, self._representation_function, np.random,
		                                                self.num_train, self.num_test, self.batch_size)


	def get_scores(self):
		return ['train_accuracy', 'eval_accuracy']



@fig.Component('task/disentanglement/beta-vae')
class BetaVAEc(DisentanglementTaskC, BetaVAE):
	def __init__(self, A, num_train=None, num_test=None, **kwargs):
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		if num_test is None:
			num_test = A.pull('num_test', 5000)
		super().__init__(A, num_train=num_train, num_test=num_test, **kwargs)



class Fairness(DisentanglementTask):
	def __init__(self, num_train=10000, num_test_points_per_class=100, **kwargs):
		super().__init__(**kwargs)

		self.num_train = num_train
		self.num_test_points_per_class = num_test_points_per_class


	def _compute(self, **kwargs):
		return fairness.compute_fairness(self.dataset, self._representation_function, np.random,
		                                 self.num_train, self.num_test_points_per_class, self.batch_size)


	def get_scores(self):
		return ['informativeness_train', 'informativeness_test', 'disentanglement', 'completeness']


	def get_results(self):
		return ['importance_matrix']



@fig.Component('task/disentanglement/fairness')
class FairnessC(DisentanglementTaskC, Fairness):
	def __init__(self, A, num_train=None, num_test_points_per_class=None, **kwargs):
		if num_train is None:
			num_train = A.pull('num_train', 10000)
		if num_test_points_per_class is None:
			num_test_points_per_class = A.pull('num_test_points_per_class', 100)
		super().__init__(A, num_train=num_train, num_test_points_per_class=num_test_points_per_class, **kwargs)



# class ResponseTask(EncoderTask, DecoderTask, ObservationTask):
# 	def __init__(self, criterion=None, **kwargs):
# 		super().__init__(**kwargs)
# 		self.criterion = criterion
#
#
# 	def _prep(self, info):
# 		info.scores = []
# 		return super()._prep(info)
#
#
# 	def _process_batch(self, info):
#
#
#
# 		pass
#
#
# 	def _apply_interventions(self, info):
#
# 		pass
#
#
# 	def response_function(self, z):
# 		with torch.no_grad():
# 			return self.encoder(self.decoder(z))



# @fig.Component('task/consistency')
# class Consistency(DecoderTask, ObservationTask, DisentanglementTask):
# 	def _prep(self, info):
#
# 		pass
#
# 	def response_function(self):
# 		pass
#
#
# 	def _process_batch(self, info):
# 		pass



#
# @fig.Component('metric/structure')
# class StructureScore(Disentanglement_Evaluator):
# 	def __init__(self, A, num_q=None, batch_size=None, **kwargs):
#
# 		if num_q is None:
# 			num_q = A.pull('num_q', 10000)
#
# 		if batch_size is None:
# 			batch_size = A.pull('batch_size', 64)
#
# 		pbar = A.pull('pbar', None)
#
# 		super().__init__(A, **kwargs)
#
# 		self.pbar = pbar
# 		self.num_q = num_q
# 		self.batch_size = batch_size
#
# 	def _compute(self, info=None):
#
# 		# run_name = info.get_name()
#
# 		model = self.model
#
# 		fullQ = []
# 		total = self.num_q
# 		bs = self.batch_size
# 		loader = self.dataset.get_loader(infinite=True, shuffle=True, seed=0, batch_size=bs)
# 		loader = iter(loader)
# 		if self.pbar is not None:
# 			self.pbar(total=total)
# 		while len(fullQ) < total // bs:
# 			batch = next(loader)
# 			x = model._process_batch(batch).original
# 			with torch.no_grad():
# 				q = model.encode(x)
# 				if isinstance(q, distrib.Distribution):
# 					q = q.sample()
# 				fullQ.append(q)
# 			if self.pbar is not None:
# 				self.pbar.update(bs)
# 		del loader
# 		fullQ = torch.cat(fullQ)
#
# 		@torch.no_grad()
# 		def response_function(q):
# 			# q = q.to(device)
# 			r = model.encode(model.decode(q))
# 			if isinstance(r, distrib.Distribution):
# 				r = r.sample()
# 			return r
#
# 		@torch.no_grad()
# 		def encode(x):
# 			with torch.no_grad():
# 				q = model.encode(x)
# 				if isinstance(q, distrib.Distribution):
# 					q = q.sample()
# 			return q
#
# 		prior = model.sample_prior(total)
# 		rprior = util.process_in_batches(response_function, prior, batch_size=self.batch_size)
#
# 		score = util.MMD(prior.cpu(), rprior.cpu()).item()
#
# 		return {'structure_score': score}, {
# 			'p-r': score,
# 			'p-q': util.MMD(prior.cpu(), fullQ.cpu()).item(),
# 			'q-r': util.MMD(fullQ.cpu(), rprior.cpu()).item(),
# 		}
#
# 	def get_results(self):
# 		return ['p-r', 'p-q', 'q-r']
#
# 	def get_scores(self):
# 		return ['structure_score']
#
#
# @fig.Component('metric/responses')
# class LatentResponses(Disentanglement_Evaluator):
# 	def __init__(self, A, **kwargs):
#
# 		num_groups = A.pull('num_groups', 50)
# 		num_q = A.pull('num_latent', 10000)
# 		num_resp = A.pull('num_response', 256)
# 		batch_size = A.pull('batch_size', 64)
#
# 		dist_type = A.pull('dist-type', 'rms')
# 		# force_different = A.pull('force-different', True)
#
# 		normalize = A.pull('normalize', True)
# 		include_q = A.pull('include-q', True)
#
# 		figure_dir = A.pull('figure_dir', None)
# 		if figure_dir is not None:
# 			figure_dir = Path(figure_dir)
# 			if not figure_dir.is_dir():
# 				figure_dir.mkdir(exist_ok=True)
#
# 		pbar = A.pull('pbar', None)
#
# 		super().__init__(A, **kwargs)
#
# 		self.figure_dir = figure_dir
#
# 		self.num_groups = num_groups
# 		self.num_q = num_q
# 		self.num_resp = num_resp
# 		self.batch_size = batch_size
# 		self.pbar = pbar
#
# 		self.dist_type = dist_type
# 		# self.force_different = force_different
# 		self.normalize = normalize
#
# 		self.include_q = include_q
#
# 		self.interventions = None
#
# 	def get_results(self):
# 		return ['response_mat', 'covariance', 'factor_responses', 'factor_responses_q']
#
# 	def get_scores(self):
# 		return ['disentanglement']
#
# 	def _compute(self, info):
#
# 		run_name = info.get_name()
#
# 		model = self.model
#
# 		fullQ = []
# 		total = self.num_q
# 		bs = self.batch_size
# 		loader = self.dataset.get_loader(infinite=True, shuffle=True, seed=0, batch_size=bs)
# 		loader = iter(loader)
# 		if self.pbar is not None:
# 			self.pbar(total=total)
# 		while len(fullQ) < total // bs:
# 			batch = next(loader)
# 			x = model._process_batch(batch).original
# 			with torch.no_grad():
# 				q = model.encode(x)
# 				if isinstance(q, distrib.Distribution):
# 					q = q.loc
# 				fullQ.append(q)
# 			if self.pbar is not None:
# 				self.pbar.update(bs)
# 		del loader
# 		fullQ = torch.cat(fullQ)
#
# 		# scales = fullQ.std(0) if self.normalize else None
#
# 		C = np.cov(fullQ.cpu().t().numpy())
# 		if self.figure_dir is not None:
# 			util.plot_mat(C, val_fmt=2)
# 			plt.tight_layout()
# 			util.save_figure(f'{run_name}_cov', root=self.figure_dir)
#
# 		R = response_mat(fullQ, model.encode, model.decode, n_interv=self.num_resp, max_batch_size=self.batch_size)
# 		# scales=scales,
# 		# dist_type='rms', force_different=True)
#
# 		if self.figure_dir is not None:
# 			util.plot_mat(R, val_fmt=1)  # responses
# 			plt.ylabel('Intervention')
# 			plt.xlabel('Response')
# 			plt.tight_layout()
# 			util.save_figure(f'{run_name}_responses', root=self.figure_dir)
#
# 			util.plot_mat((R @ R.t()), val_fmt=1)  # interactions?
# 			plt.tight_layout()
# 			util.save_figure(f'{run_name}_interactions', root=self.figure_dir)
#
# 		sampler = info.get_config().pull('sampler', None)
# 		if sampler is None:
# 			sampler = InterventionSamplerBase(self.dataset)
# 		else:
# 			self.interventions = None
#
# 		try:
# 			if self.interventions is None:
# 				self.interventions = sample_full_interventions(sampler, num_groups=self.num_groups, pbar=self.pbar)
# 		except:
# 			raise
# 			print('Skipping factor responses')
#
# 			return {}, \
# 			       {'response_mat': R, 'covariance': C, }
#
# 		out = conditioned_reponses(model.encode, model.decode, self.interventions, pbar=self.pbar,
# 		                           include_q=self.include_q,
# 		                           resp_kwargs=dict(max_batch_size=self.batch_size,
# 		                                            # scales=scales,
# 		                                            # force_different=self.force_different
# 		                                            ))
#
# 		if self.include_q:
# 			mats, lts = out
# 		else:
# 			mats, lts = out, None
#
# 		M = mats.min(1)[0].max(-1)[0]
# 		if self.figure_dir is not None:
# 			factors = self.dataset.get_factor_order()
# 			util.plot_mat(M, val_fmt=1)
# 			plt.yticks(range(len(factors)), factors)
# 			plt.xlabel('Latent dimension')
# 			plt.tight_layout()
# 			util.save_figure(f'{run_name}_factor-responses', root=self.figure_dir)
#
# 			try:
# 				graph = self.dataset.get_adjacency_matrix()
# 			except AttributeError:
# 				pass
# 			else:
# 				util.plot_mat(graph, val_fmt=1)
# 				plt.yticks(range(len(factors)), factors)
# 				plt.xlabel('True Dimension')
# 				plt.tight_layout()
# 				util.save_figure(f'{run_name}_graph', root=self.figure_dir)
#
# 		disentanglement = M.max(0)[0].sum() / M.sum()
#
# 		return {'disentanglement': disentanglement}, \
# 		       {'response_mat': R, 'covariance': C, 'factor_responses': mats, 'factor_responses_q': lts, }




