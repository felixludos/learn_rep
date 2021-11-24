from pathlib import Path
import numpy as np
import torch
import h5py as hf

from omnibelt import unspecified_argument, get_printer
import omnifig as fig

prt = get_printer(__file__)

try:
	import faiss
except ImportError:
	prt.warning('Failed to import faiss')


from omnilearn import util
from omnilearn.data import DataLike
from omnilearn.op import get_save_dir, framework as fm
from omnilearn.models import get_loss_type

from .common import GeneratorTask, GeneratorTaskC, ExtractionTask, ExtractionTaskC, \
	IterativeTask, IterativeTaskC, EncoderTask, EncoderTaskC



# @fig.Script('reformat-fid')
# def _reformat_fid(A):
# 	base_props = {'ID': 'inception-v3', 'pretrained': True}
#
# 	dataset = fig.run('load-data', A)
#
# 	root = dataset.get_root()
# 	name = f'{dataset.cat}_fid_stats.h5'
#
# 	f = hf.File(root / name)
# 	list(f.keys())
#
# 	for key in [key for key in f.keys() if key.endswith('_mu')]:
# 		mode, dim, _ = key.split('_')
# 		dim = int(dim)
# 		props = {'dim': dim, 'mode': mode, **base_props}
#
# 		dataset.save_stats(props, (torch.from_numpy(f[f'{mode}_{dim}_mu'][:]),
# 		                           torch.from_numpy(f[f'{mode}_{dim}_sigma'][:])))
#
# 	table = dataset.get_datafile('stats/table')
#
# 	print(table)
#
# 	pass


class GenerationTask(GeneratorTask, ExtractionTask, IterativeTask):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		if self.extractor is None:
			prt.warning('No feature extractor for the generative task')
		self._gen_fn = None


	def _get_title(self):
		ref = f' (reference)' if self.generator is None else ''
		return f'Evaluating Generative Task{ref}'


	def _prep(self, info):
		if self.generator is None:
			self._gen_fn = self.dataset.to_loader(infinite=True, sample_format='observation',
			                                      batch_size=self.batch_size).demand
		else:
			self._gen_fn = self.generator.generate

		info.features = []

		return super()._prep(info)


	def _generate_batch(self, info):
		if 'feats' in info:
			del info.feats
		info.samples = self._gen_fn(info.num)
		return super()._generate_batch(info)


	def _process_batch(self, info):
		if 'feats' not in info:
			info.feats = info.samples if self.extractor is None else self.extractor.encode(info.samples)
		info.features.append(info.feats.cpu()) # a little hacky
		return super()._process_batch(info)


	def _aggregate_results(self, info):
		info.stats = self._compute_stats(info)
		if self.generator is not None:
			info = self._compare_to_reference(info)
		if 'features' in info and self._slim:
			del info.features
		if 'referernce' in info and self._slim:
			del info.reference
		return super()._aggregate_results(info)


	def _compute_reference(self, props):
		_gen = self.generator
		self.generator = None
		ref_stats = self._run().stats
		self.generator = _gen
		return ref_stats



class DiscriminatorGenerationTask(GenerationTask):
	pass



class AggregateGenerationTask(GenerationTask):
	pass



@fig.Component('task/generation')
class GenerationTaskC(GeneratorTaskC, ExtractionTaskC, IterativeTaskC, GenerationTask):
	pass



class NoveltyTask(GeneratorTask, EncoderTask, IterativeTask):
	def __init__(self, metric='l2', num_nearest=1, **kwargs):
		super().__init__(**kwargs)
		if metric != 'l2':
			raise NotImplementedError

		self.index = None
		self.metric = get_loss_type(metric)
		self.num_nearest = num_nearest


	def _prep(self, info):
		self.index = faiss.IndexFlatL2(self.dataset.din)
		self.index.add(util.combine_dims(self.dataset.get('observations')))

		info.distances = []


	def _generate_batch(self, info):
		if 'feats' in info:
			del info.feats
		info.samples = self.generator.generate(info.num)
		return super()._generate_batch(info)


	def _process_batch(self, info):
		if 'feats' not in info:
			info.feats = util.combine_dims(info.samples) if self.encoder is None \
				else self.encoder.encode(info.samples)

		info.dists, info.nearest = self.find_nearest(info.feats, self.num_nearest)
		info.distances.append(info.dists)
		return super()._process_batch(info)


	def find_nearest(self, samples, k=1):
		return self.index.search(samples, k)


	def _aggregate_results(self, info):
		info.distances = torch.cat(info.distances)
		info.score = info.distances.mean()

		if self._slim:
			del info.distances
		return super()._aggregate_results(info)



@fig.Component('task/novelty')
class NoveltyTaskC(GeneratorTaskC, EncoderTaskC, IterativeTaskC, NoveltyTask):
	def __init__(self, A, metric=unspecified_argument, num_nearest=None, **kwargs):

		if metric is unspecified_argument:
			metric = A.pull('metric', 'l2', ref=True)

		if num_nearest is None:
			num_nearest = A.pull('num_nearest', 1, ref=True)

		super().__init__(A, metric=metric, num_nearest=num_nearest, **kwargs)



