from pathlib import Path
import numpy as np
import torch

from omnibelt import unspecified_argument, get_printer
import omnifig as fig

from omnilearn import util
from omnilearn.data import DataLike
from omnilearn.op import get_save_dir, framework as fm
from omnilearn.models import get_loss_type

import matplotlib.pyplot as plt

prt = get_printer(__file__)

from .common import ObservationTask, ObservationPairTask, IterativeTaskC, EncoderTask, EncoderTaskC, \
	ExtractionTask, ExtractionTaskC, DecoderTask, DecoderTaskC, EncodedObservationTask

from .compat import BitsBackCompressor


class ReconstructionTask(EncodedObservationTask, DecoderTask): # TODO: make sure images are bytes when needed
	def __init__(self, criterion='ms-ssim', compressor=None, **kwargs):
		super().__init__(**kwargs)
		self.criterion = get_loss_type(criterion)
		self.compressor = compressor


	def get_scores(self):
		return ['score', 'bpp']


	def get_results(self):
		return ['scores', 'bits']


	def _prep(self, info):
		info.scores = []
		info.bits = []
		return super()._prep(info)


	def _rec_scores(self, info):
		return self.criterion(info.reconstructions, info.targets if 'targets' in info else info.originals)


	def _process_batch(self, info):
		if self.compressor is not None:
			bits = self.compressor.compress(info.observations)
			info.compressed = bits
			info.bits.append(8*len(bits) / len(info.originals)) # average over batch
		info.reconstructions = self.decoder.decode(info.observations)
		info.scores.append(self._rec_scores(info))
		return super()._process_batch(info)


	def _aggregate_results(self, info):
		info.scores = torch.as_tensor(info.scores)
		info.score = info.scores.mean().item()

		if len(info.bits):
			info.bits = torch.as_tensor(info.bits)
			info.bpd = info.bits.mean().item() / info.originals[0].numel()
		else:
			del info.bits

		if self._slim:
			del info.scores
			if 'bits' in info:
				del info.bits
		return info



@fig.Component('task/reconstruction')
class ReconstructionTaskC(IterativeTaskC, EncoderTaskC, DecoderTaskC, ReconstructionTask):
	def __init__(self, A, criterion=unspecified_argument, compressor=unspecified_argument, **kwargs):
		if criterion is unspecified_argument:
			criterion = A.pull('criterion', 'ms-ssim', ref=True)
		if compressor is unspecified_argument:
			compressor = A.pull('compressor', None, ref=True)
		super().__init__(A, criterion=criterion, compressor=compressor, **kwargs)



class MetricTask(ObservationPairTask, EncodedObservationTask):
	def __init__(self, metric='l2', criterion='cosine-similarity', sample_format={'observations', 'labels'}, **kwargs):
		super().__init__(sample_format=sample_format, **kwargs)
		self.metric = get_loss_type(metric)
		self.criterion = get_loss_type(criterion)


	def get_scores(self):
		return ['score']


	def get_results(self):
		return ['distances', 'trues']


	def _prep(self, info):
		info.distances = []
		info.trues = []


	def _split_batch(self, info):
		info = super()._split_batch(info)
		info.a_labels, info.b_labels = info.labels.split(2)
		return info


	def _process_batch(self, info):
		info.distances.append(self.metric.distance(info.a, info.b).squeeze())
		info.trues.append(self.dataset.distance(info.a_labels, info.b_labels).squeeze())
		return super()._process_batch(info)


	def _aggregate_results(self, info):
		info.distances = torch.cat(info.distances, 0).unsqueeze(0)
		info.trues = torch.cat(info.trues, 0).unsqueeze(0)

		info.score = self.criterion(info.distances, info.trues)
		if self._slim:
			del info.distances, info.trues
		return info



@fig.Component('task/metric')
class MetricTaskC(IterativeTaskC, MetricTask):
	def __init__(self, A, metric=unspecified_argument, criterion=unspecified_argument, **kwargs):

		if metric is unspecified_argument:
			metric = A.pull('metric', 'l2', ref=True)

		if criterion is unspecified_argument:
			criterion = A.pull('criterion', 'cosine-similarity', ref=True)

		super().__init__(A, metric=metric, criterion=criterion, **kwargs)



class InterpolationTask(ObservationPairTask, EncodedObservationTask, DecoderTask, ExtractionTask):
	def __init__(self, interpolator=None, num_steps=12, batch_size=12, **kwargs):
		super().__init__(batch_size=batch_size, **kwargs)
		self.interpolator = interpolator

		self._num_steps = num_steps


	def _prep(self, info):

		info.interps = []
		info.ends = []

		return super()._prep(info)


	def _process_batch(self, info):

		a, b = info.a, info.b

		if self.interpolator is None: # linearly interpolate
			a, b = a.unsqueeze(1), b.unsqueeze(1)
			progress = torch.linspace(0., 1., steps=self._num_steps, device=a.device)\
				.view(1, self._num_steps, *[1]*len(a.shape[2:]))
			info.steps = a + (b-a)*progress

		else:
			info.steps = self.interpolator(a, b, self._num_steps)

		info.steps = util.combine_dims(info.steps, 0, 1)
		info.samples = info.steps if self.decoder is None else self.decoder(info.steps)

		if 'stats' in info:
			del info.stats
		if 'features' in info:
			del info.features
		info = self._compare_to_reference(info)
		scores = info.score
		del info.score

		interp = scores[:,1:-1]
		ends = scores[:, [0,-1]]

		info.interps.append(interp)
		info.ends.append(ends)

		return info


	def _compute_reference(self, props):
		raise NotImplementedError # TODO: handle this for encoded datasets
		return GenerationTask(dataset=self.dataset, extractor=self.extractor, n_samples=self.n_samples,
		                      batch_size=self._num_steps*self.batch_size).compute().stats


	def _aggregate_results(self, info):

		info.interps = torch.cat(info.interps)
		info.ends = torch.cat(info.ends)

		info.score = info.ends.mean() / info.interps.mean()

		if self._slim:
			del info.interps, info.ends
		else:
			info.steps = util.split_dim(info.steps, -1, self._num_steps)
			info.paths = util.split_dim(info.samples, -1, self._num_steps)
			del info.samples
			info.features = util.split_dim(info.features, -1, self._num_steps)

		return super()._aggregate_results(info)



@fig.Component('task/interpolation')
class InterpolationTaskC(IterativeTaskC, EncoderTaskC, DecoderTaskC, ExtractionTaskC, InterpolationTask):
	def __init__(self, A, interpolator=None, num_steps=None, **kwargs):

		if interpolator is unspecified_argument:
			interpolator = A.pull('interpolator', None, ref=True)

		if num_steps is None:
			num_steps = A.pull('num_steps', 12)

		super().__init__(A, interpolator=interpolator, num_steps=num_steps, **kwargs)



class LosslessCompressionTask(ObservationTask, EncoderTask, DecoderTask):
	def get_scores(self):
		return ['score', 'bpd']


	def get_results(self):
		return ['counts', 'bytes']


	def _prep(self, info):
		info.counts = []

		self.compressor = BitsBackCompressor(self.encoder, self.decoder, seed=5)
		# stream = self.compressor.generate_seed_state()
		# self.compressor.set_state(stream)
		# info.start_stream = self.compressor.state_to_bytes(stream)

		return super()._prep(info)


	def _process_batch(self, info):
		info.bytes = self.compressor.compress(info.observations)
		info.counts.append(8*len(info.bytes) / info.observations.numel())

		# counts = []
		# past = 0
		# state = self.compressor.generate_seed_state()
		# info.initial_stream = state

		# z = self.compressor.compress(info.observations)
		# xhat = self.compressor.decompress(z)

		# for sample in info.observations:
		# 	state = self.compressor.compress_append([sample], state)
		# 	total = self.compressor.count_bits(state)
		# 	counts.append(total - past)
		# 	past = total
		# info.counts.extend(counts)
		#
		# info.state = state
		return super()._process_batch(info)


	def _aggregate_results(self, info):
		info.counts = torch.as_tensor(info.counts)
		info.bpd = info.counts.float().mean().item()
		# mean_bits = info.counts.float().mean()
		# info.bpd = mean_bits / info.observations[0].numel()
		info.score = 1. -  info.bpd / 8.

		# info.initial_stream = self.compressor.state_to_bytes(info.initial_stream)
		# info.final_stream, info.uncompressed = self.compressor.partial_decompress(info.state)
		# info.final_stream = self.compressor.state_to_bytes(info.final_stream)
		# info.bytes = self.compressor.state_to_bytes(info.state)

		info.reconstructions = self.compressor.decompress(info.bytes)
		if self._slim:
			del info.counts
		return info



@fig.Component('task/lossless-compression')
class LosslessCompressionTaskC(IterativeTaskC, EncoderTaskC, DecoderTaskC, LosslessCompressionTask):
	pass





