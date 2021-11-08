from pathlib import Path
from itertools import chain
import numpy as np
import torch

from omnibelt import unspecified_argument, get_printer
import omnifig as fig

from omnilearn import util
from omnilearn.data import DataLike, DatasetWrapper, Dataset
from omnilearn.op import get_save_dir, framework as fm


from .common import Task, TaskC



class GeneralizationTask(Task):
	def __init__(self, task, dataset=None, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.task = task


	def _get_replacements(self, **kwargs):
		return kwargs


	def get_scores(self):
		return self.task.get_scores()


	def get_results(self):
		return self.task.get_results()


	def _compute(self, **kwargs):
		return self.task._compute(**self._get_replacements(**kwargs))



class GeneralizationTaskC(TaskC, GeneralizationTask):
	def __init__(self, A, task=None, **kwargs):
		if task is None:
			task = A.pull('task', None)
		super().__init__(A, task=task, **kwargs)



class ComparisonTask(GeneralizationTask):
	def __init__(self, original_results=None, **kwargs):
		super().__init__(**kwargs)
		self._original_results = original_results
		self._evaluating_updated = None

	def _get_originals(self, **kwargs):
		return kwargs


	def get_scores(self):
		base = self.task.get_scores()
		scores = ['score'] if 'score' in base else []
		scores.extend(chain([f'original-{key}', f'updated-{key}'] for key in base))
		return scores


	def get_results(self):
		return list(chain([f'original-{key}', f'updated-{key}'] for key in  self.task.get_scores()))


	def _compute(self, original_results=None, **kwargs):
		if original_results is not None:
			self._original_results = original_results
		if self._original_results is None:
			self._evaluating_updated = False
			self._original_results = self.task._compute(**self._get_originals(**kwargs))
		self._evaluating_updated = True
		updated = self.task._compute(**self._get_replacements(**kwargs))
		self._evaluating_updated = None
		return self._merge_results(self._original_results, updated)


	def _merge_results(self, original, updated):
		merged = util.TensorDict()

		for key, val in original.items():
			merged[f'original-{key}'] = val
		for key, val in updated.items():
			merged[f'updated-{key}'] = val

		if 'score' in original and 'score' in updated:
			merged.score = updated.score / original.score
		return merged



class ComparisonTaskC(GeneralizationTaskC, ComparisonTask):
	def __init__(self, A, original_results=unspecified_argument, **kwargs):
		if original_results is None:
			original_results = A.pull('original-results', None)
		super().__init__(A, original_results=original_results, **kwargs)



@DatasetWrapper('batch-augmentation')
class BatchAugmentation(Dataset):
	def __init__(self, A, augmentation=unspecified_argument, **kwargs):

		if augmentation is unspecified_argument:
			augmentation = A.pull('augmentation', None)
		super().__init__(A, **kwargs)
		BatchAugmentation.__activate__(augmentation=augmentation)


	def __activate__(self, augmentation=None):
		self.augmentation = augmentation


	def get(self, name=None, idx=None, format=None, **kwargs):
		batch = super().get(name=name, idx=idx, format=format, **kwargs)
		if name is None and self.augmentation is not None:
			batch = self.augmentation(batch, idx=idx, format=format, **kwargs)
		return batch



class TransferTask(ComparisonTask):
	def __init__(self, augmentation, invariance=False, **kwargs):
		super().__init__(**kwargs)
		self.augmentation = augmentation
		self._invariance = invariance


	def _augment_batch(self, batch, idx=None, format=None, **kwargs):
		if self._evaluating_updated and self.augmentation is not None:
			batch.unaugmented = batch.observations
			batch.observations = self.augmentation(batch.observations)
			if self._invariance:
				batch.targets = batch.unaugmented
		return batch


	def _get_replacements(self, dataset=None, **kwargs):
		kwargs = super()._get_replacements(**kwargs)
		if dataset is None:
			dataset = self.task.dataset
			dataset.register_wrapper('batch-augmentation', augmentation=self._augment_batch)
		kwargs['dataset'] = dataset
		return kwargs



@fig.Component('tasks/transfer')
class TransferTaskC(ComparisonTaskC, TransferTask):
	def __init__(self, A, augmentation=unspecified_argument, invariance=None, **kwargs):
		if augmentation is unspecified_argument:
			augmentation = A.pull('augmentation', None)
		if invariance is None:
			invariance = A.pull('invariance', False)
		super().__init__(A, augmentation=augmentation, invariance=invariance, **kwargs)



class QuantizationTask(ComparisonTask):
	def __init__(self, quantizer, encoder, **kwargs):
		super().__init__(**kwargs)

		self.quantizer = quantizer
		self.encoder = encoder
		self.dout = getattr(encoder, 'latent_dim', getattr(encoder, 'dout'))


	def _get_originals(self, encoder=None, **kwargs):
		kwargs = super()._get_replacements(**kwargs)
		if encoder is None:
			encoder = self.encoder
		kwargs['encoder'] = encoder
		return kwargs


	def _get_replacements(self, encoder=None, **kwargs):
		kwargs = super()._get_replacements(**kwargs)
		if encoder is None:
			encoder = self
		kwargs['encoder'] = encoder
		return kwargs


	def encode(self, x):
		z = self.encoder.encode(x)
		return self.quantizer.quantize(z)



@fig.Component('task/quantization')
class QuantizationTaskC(ComparisonTaskC, QuantizationTask):
	def __init__(self, A, quantizer=unspecified_argument, encoder=unspecified_argument, **kwargs):
		if quantizer is unspecified_argument:
			quantizer = A.pull('quantizer', None)
		if encoder is unspecified_argument:
			encoder = A.pull('encoder', None)
		super().__init__(A, quantizer=quantizer, encoder=encoder, **kwargs)



class ExtrapolationTask(ComparisonTask):
	def __init__(self, original_dataset, extended_dataset, **kwargs):
		super().__init__(**kwargs)
		self.original_dataset = original_dataset
		self.extended_dataset = extended_dataset


	def _get_originals(self, dataset=None, **kwargs):
		kwargs = super()._get_originals(**kwargs)
		if dataset is None:
			dataset = self.original_dataset
		kwargs['dataset'] = dataset
		return kwargs


	def _get_replacements(self, dataset=None, **kwargs):
		kwargs = super()._get_replacements(**kwargs)
		if dataset is None:
			dataset = self.extended_dataset
		kwargs['dataset'] = dataset
		return kwargs



@fig.Component('task/extrapolation')
class ExtrapolationTaskC(ComparisonTaskC, ExtrapolationTask):
	def __init__(self, A, original_dataset=None, extended_dataset=None, **kwargs):
		if original_dataset is None:
			original_dataset = A.pull('original_dataset', '<>dataset', ref=True)
		if extended_dataset is None:
			extended_dataset = A.pull('extended_dataset', '<>dataset', ref=True)
		super().__init__(A, original_dataset=original_dataset, extended_dataset=extended_dataset, **kwargs)


