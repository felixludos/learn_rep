from pathlib import Path

from omnibelt import unspecified_argument, get_printer
import omnifig as fig

import numpy as np
import torch

# import gym
# from gym.spaces import Discrete, MultiDiscrete

import torch
# from torch import distributions as distrib
# from torch.utils.data import DataLoader, TensorDataset



# from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from omnilearn import util

from omnilearn.models import get_loss_type
from omnilearn.op import get_save_dir, framework as fm#, scikit as sk
from omnilearn.op import Run


prt = get_printer(__file__)


@fig.Script('eval-tasks')
def run_tasks(config, run=None, tasks=None, overwrite=None, use_dataset=None):
	if run is None:
		run = fig.run('load-run', config)
	# return run.eval_tasks(config=config)

	if overwrite is None:
		overwrite = config.pull('overwrite', False)

	if use_dataset is None and config is not None:
		use_dataset = config.pull('use-dataset', False)

	if tasks is None:
		tasks = config.pull('tasks')

	names = ', '.join(tasks.keys())
	print(f'Evaluating {len(tasks)} tasks: {names}')
	if overwrite:
		print('Overwriting past results')

	for name, task in tasks.items():
		if not run.has_datafile(f'tasks/{name}') or overwrite:
			kwargs = {}
			if use_dataset:
				kwargs['dataset'] = run.get_dataset()
			scores, results = task.compute(**kwargs)
			run.update_datafile(f'tasks/{name}', {'scores':scores, 'results':results})
			score = f' (score: {results.score:.3g})' if 'score' in results else ''
			print(f'{name} complete{score}')

	print('All tasks complete')



@fig.AutoModifier('tasked')
class Tasked(Run):

	def eval_tasks(self, tasks=None, overwrite=None, use_dataset=None):
		return run_tasks(config=self.get_config(), tasks=tasks, overwrite=overwrite, use_dataset=use_dataset)
		# if overwrite is None and config is not None:
		# 	overwrite = config.pull('overwrite', False)
		#
		# if use_dataset is None and config is not None:
		# 	use_dataset = config.pull('use-dataset', False)
		#
		# if tasks is None:
		# 	if config is not None:
		# 		tasks = config.pull('tasks')
		# 	else:
		# 		tasks = self.tasks
		#
		# assert tasks is not None
		#
		# names = ', '.join(tasks.keys())
		# print(f'Evaluating {len(tasks)} tasks: {names}')
		# if overwrite:
		# 	print('Overwriting past results')
		#
		# for name, task in tasks.items():
		# 	if not self.has_datafile(f'tasks/{name}') or overwrite:
		# 		kwargs = {}
		# 		if use_dataset:
		# 			kwargs['dataset'] = self.get_dataset()
		# 		results = task.compute(**kwargs)
		# 		self.update_datafile(f'tasks/{name}', results)
		# 		score = f' (score: {results.score:.3g})' if 'score' in results else ''
		# 		print(f'{name} complete{score}')
		#
		# print('All tasks complete')



# @fig.Script('test-estimator')
# def _test_estimator(A):
#
# 	seed = A.pull('seed', util.gen_random_seed())
#
# 	dataset = fig.run('load-data', A)
# 	# dataset.register_wrapper('subset', kwargs={'num': 100, 'update_data': False})
#
# 	# model = fig.run('load-model', A)
# 	# dataset.set_encoder(model)
#
# 	space = dataset.get_target_space()
# 	print(space)
#
# 	builder = A.pull('estimator')
# 	est = builder.build(dataset)
#
# 	# obs = dataset.get('observations')
# 	# print(obs.shape, )
#
# 	# out = est.fit(dataset)
#
# 	# pred = est.predict(obs)
# 	# y = dataset.get('targets')
#
# 	dataset.switch_to('val')
#
# 	out = est.compute(dataset=dataset)
#
# 	# print(pred[:4])
# 	# print(y[:4])
#
# 	print(out.keys())
#
# 	pass

# from .lossy_compression.ppm_compress import compress
# from .lossy_compression.ppm_decompress import decompress

# import io
# from collections import Counter
# from .lossy_compression import arithmeticcoding
# # from .lossy_compression.adaptive_arithmetic_compress import compress
# # from .lossy_compression.adaptive_arithmetic_decompress import decompress
#
# from src.tasks.lossy_compression.arithmetic_compress import compress
# from src.tasks.lossy_compression.arithmetic_decompress import decompress

# @fig.Script('test-compression')
# def _test_compression(A):
#
# 	np.random.seed(5)
# 	X = np.random.randn(20) * 10 > 0
# 	X = X.astype(np.float32).astype(np.int8)
#
# 	x = X.tobytes()
#
# 	cnts = Counter(x)
# 	cnts[256] = 1
# 	ls = [0] * (max(cnts) + 1)
# 	for i, v in cnts.items():
# 		ls[i] = v
# 	freqs = arithmeticcoding.SimpleFrequencyTable(ls)
#
# 	cinp = io.BytesIO(x)
# 	cout = io.BytesIO()
# 	bitout = arithmeticcoding.BitOutputStream(cout)
#
# 	compress(freqs, cinp, bitout)
#
# 	z = cout.getvalue()
# 	print(len(x), len(z), len(x) / len(z))
#
# 	dinp = io.BytesIO(z)
# 	bitin = arithmeticcoding.BitInputStream(dinp)
# 	dout = io.BytesIO()
#
# 	decompress(freqs, bitin, dout)
#
# 	xhat = dout.getvalue()
# 	print(x == xhat[:len(x)], len(x), len(xhat))
# 	print('done')





class Task(fm.Computable): # TODO: specify random seed for reproducibility
	def __init__(self, dataset, slim=False, score_key=None, **kwargs):
		super().__init__(**kwargs)
		self.dataset = dataset
		self._slim = slim
		self._score_key = score_key


	def required_modules(self):
		return []


	def get_scores(self):
		return ['score', *super().get_scores()]


	def _compute(self, dataset=unspecified_argument, **kwargs):
		if dataset is not unspecified_argument:
			self.dataset = dataset
		out = self._run(**kwargs)
		if 'score' not in out and self._score_key is not None:
			out['score'] = out[self._score_key]
		return out


	def _run(self, out=None):
		if out is None:
			out = util.TensorDict()
		return out



class TaskC(fig.Configurable, Task):
	def __init__(self, A, dataset=unspecified_argument, slim=None, score_key=unspecified_argument, **kwargs):
		if dataset is unspecified_argument:
			dataset = A.pull('dataset', None, ref=True)
		if slim is None:
			slim = A.pull('slim', False)
		if score_key is unspecified_argument:
			score_key = A.pull('score-key', None)
		super().__init__(A, dataset=dataset, slim=slim, score_key=score_key, **kwargs)



class EncoderTask(Task):
	def __init__(self, encoder=None, use_distribution=False, **kwargs):
		super().__init__(**kwargs)
		self.encoder = encoder
		self._use_distribution = use_distribution


	def required_modules(self):
		return ['encoder', *super().required_modules()]


	def _compute(self, encoder=unspecified_argument, **kwargs):
		if encoder is not unspecified_argument:
			self.encoder = encoder
		return super()._compute(**kwargs)



class EncoderTaskC(TaskC, EncoderTask):
	def __init__(self, A, encoder=unspecified_argument, use_distribution=None, **kwargs):
		if encoder is unspecified_argument:
			encoder = A.pull('encoder', None, ref=True)
		if use_distribution is None:
			use_distribution = A.pull('use-distribution', False)
		super().__init__(A, encoder=encoder, use_distribution=use_distribution, **kwargs)



class DownstreamTask(EncoderTask):
	def __init__(self, auto_wrap=True, **kwargs):
		super().__init__(**kwargs)
		self._auto_wrap = auto_wrap


	def _run(self, out=None):
		out = super()._run(out=out)
		if self._auto_wrap and self.encoder is not None:
			self.dataset.register_wrapper('encoded', encoder=self.encoder)
		return out



class DecoderTask(Task):
	def __init__(self, decoder=None, **kwargs):
		super().__init__(**kwargs)
		self.decoder = decoder


	def required_modules(self):
		return ['decoder', *super().required_modules()]


	def _compute(self, decoder=unspecified_argument, **kwargs):
		if decoder is not unspecified_argument:
			self.decoder = decoder
		return super()._compute(**kwargs)



class DecoderTaskC(TaskC, DecoderTask):
	def __init__(self, A, decoder=unspecified_argument, **kwargs):
		if decoder is unspecified_argument:
			decoder = A.pull('decoder', None, ref=True)
		super().__init__(A, decoder=decoder, **kwargs)



class MissingReferenceError(Exception):
	def __init__(self, props):
		super().__init__(f'Could not find reference stats with the correct properties (set "compute_missing_reference" '
		                 f'to automatically compute')
		self.props = props



class ExtractionTask(Task):
	def __init__(self, criterion, extractor=None,
	             compute_missing_reference=True, reference_props={}, **kwargs):
		super().__init__(**kwargs)
		self.extractor = extractor
		self.criterion = get_loss_type(criterion)
		self._compute_missing_reference = compute_missing_reference
		self._base_reference_props = reference_props


	def required_modules(self):
		return ['extractor', 'criterion', *super().required_modules()]


	def get_scores(self):
		return ['score']


	def get_results(self):
		return ['stats', 'reference']


	def _get_reference_props(self):
		props = self._base_reference_props.copy()
		if self.extractor is not None:
			props.update(self.extractor.get_hparams())
		return props


	def find_reference_stats(self, props=None):
		if props is None:
			props = self._get_reference_props()
		try:
			return self.dataset.load_stats(props)
		except FileNotFoundError:
			if self._compute_missing_reference:
				stats = self._compute_reference(props)
				self.dataset.save_stats(props, stats)
				return stats
			raise MissingReferenceError(props)


	def _compute_reference(self, props):
		raise NotImplementedError


	def _compute(self, extractor=unspecified_argument, criterion=unspecified_argument, **kwargs):
		if extractor is not unspecified_argument:
			self.extractor = extractor
		if criterion is not unspecified_argument:
			self.criterion = criterion
		return super()._compute(**kwargs)


	def _compare_to_reference(self, info):
		if 'stats' not in info:
			if 'features' not in info:
				assert 'samples' in info
				info.features = info.samples if self.extractor is None else self.extractor.encode(info.samples)
				info.features = info.features.cpu() # a little hacky
			info.stats = self._compute_stats(info)
		assert 'stats' in info, 'no stats to compare to reference'
		if 'reference' not in info:
			info.reference = self.find_reference_stats()
		info.score = self.criterion(info.stats, info.reference)
		return info


	def _compute_stats(self, info):
		assert 'features' in info
		if isinstance(info.features, list):
			info.features = torch.cat(info.features)
		return info.features.mean(0), util.cov(info.features)



class ExtractionTaskC(TaskC, ExtractionTask):
	def __init__(self, A, criterion=unspecified_argument, extractor=unspecified_argument,
	             # aggregator=unspecified_argument,
	             reference_props=None, compute_missing_reference=None, **kwargs):

		if criterion is unspecified_argument:
			criterion = A.pull('criterion', None, ref=True)
		if extractor is unspecified_argument:
			extractor = A.pull('extractor', None, ref=True)
		# if aggregator is unspecified_argument:
		# 	aggregator = A.pull('aggregator', None, ref=True)

		if reference_props is None:
			reference_props = A.pull('reference-props', {})
		if compute_missing_reference is None:
			compute_missing_reference = A.pull('compute_missing_reference', True)

		super().__init__(A, criterion=criterion, extractor=extractor,
		                 # aggregator=aggregator,
		                 compute_missing_reference=compute_missing_reference, reference_props=reference_props,
		                 **kwargs)



class GeneratorTask(Task):
	def __init__(self, generator=None, **kwargs):
		super().__init__(**kwargs)
		self.generator = generator


	def required_modules(self):
		return ['generator', *super().required_modules()]


	def _compute(self, generator=unspecified_argument, **kwargs):
		if generator is not unspecified_argument:
			self.generator = generator
		return super()._compute(**kwargs)



class GeneratorTaskC(TaskC, EncoderTask):
	def __init__(self, A, generator=unspecified_argument, **kwargs):
		if generator is unspecified_argument:
			generator = A.pull('generator', None, ref=True)
		super().__init__(A, generator=generator, **kwargs)



class IterativeTask(Task):
	def __init__(self, batch_size=60, n_samples=15120, pbar=None, use_dataset_len=False, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size
		self.n_samples = n_samples
		self.pbar = pbar
		self._use_dataset_len = use_dataset_len


	def _get_title(self):
		return None


	def _prep(self, info):
		if self._use_dataset_len:
			self.n_samples = len(self.dataset)
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



class IterativeTaskC(TaskC, IterativeTask):
	def __init__(self, A, batch_size=None, n_samples=None, pbar=unspecified_argument, use_dataset_len=None, **kwargs):

		if n_samples is None:
			n_samples = A.pull('n_samples', 15120)

		if batch_size is None:
			batch_size = A.pull('batch_size', 60)

		if pbar is unspecified_argument:
			pbar = A.pull('pbar', None)

		if use_dataset_len is None:
			use_dataset_len = A.pull('use_dataset_len', False)

		super().__init__(A, n_samples=n_samples, batch_size=batch_size, use_dataset_len=use_dataset_len,
		                 pbar=pbar, **kwargs)



class ObservationTask(IterativeTask):
	def __init__(self, sample_format={'observations'}, **kwargs):
		super().__init__(**kwargs)
		self._sample_format = sample_format


	def _prep(self, info):
		self._dataloader = self.dataset.to_loader(infinite=True, sample_format=self._sample_format,
		                                          batch_size=self.batch_size)
		return super()._prep(info)


	def _generate_batch(self, info):
		info = super()._generate_batch(info)
		batch = self._dataloader.demand(info.num)
		info.update(batch)
		return info


	def _aggregate_results(self, info):
		self._dataloader = None
		return super()._aggregate_results(info)



class EncodedObservationTask(ObservationTask, EncoderTask):
	def _generate_batch(self, info):
		if 'originals' in info:
			del info['originals']
		info = super()._generate_batch(info)
		if 'originals' not in info and 'observations' in info:
			info.originals = info.observations
			if self.encoder is not None:
				info.observations = self.encode(info.observations)
				if isinstance(info.observations, util.Distribution) and not self._use_distribution:
					info.observations = info.observations.bsample()
		return info


	def encode(self, samples):
		return self.encoder.encode(samples)



class ObservationPairTask(ObservationTask):


	def _generate_batch(self, info):
		info.num *= 2
		info = super()._generate_batch(info)
		info.num //= 2
		self._split_batch(info)
		return info


	def _split_batch(self, info):
		info.a, info.b = info.observations.chunk(2)
		return info



