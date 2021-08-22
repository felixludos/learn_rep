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
from omnilearn.op import get_save_dir, framework as fm
from omnilearn.eval import Evaluator
from omnilearn.data import Memory_Dataset



@fig.AutoModifier('encoded')
class Encoded(Memory_Dataset):
	def __init__(self, A, encoder=unspecified_argument, sample_best=None, batch_size=None, **kwargs):
		
		if encoder is unspecified_argument:
			encoder = A.pull('encoder', '<>model', None, ref=True)
		
		if sample_best is None:
			sample_best = A.pull('sample-best', True)
		
		if batch_size is None:
			batch_size = A.pull('batch-size', 64)
		
		super().__init__(A, **kwargs)
		
		self.encoder = encoder
		self._sample_best = sample_best
		self._batch_size = batch_size
		
		self._process_observations(A)
	
	
	def _process_observations(self, A):
		
		loader = DataLoader(TensorDataset(self.get_observations()), batch_size=self._batch_size)
		
		pbar = A.pull('progress-bar', None)
		pbar.set_description('Encoding observations')
		if pbar is not None:
			loader = pbar(loader)
		
		Q = []
		with torch.no_grad():
			for x in loader:
				q = self._encode_observation(x)
				Q.append(q)
		Q = torch.cat(Q)
		
		self._replace_observations(Q)
		
	
	def _encode_observation(self, x, **kwargs):
		q = self.encoder.encode(x, **kwargs)
		if isinstance(q, distrib.Distribution):
			q = q.bsample() if self._sample_best else q.rsample()
		return q



class Task(Evaluator, fm.Fitable):
	def __init__(self, A, dataset=unspecified_argument, **kwargs):
		
		if dataset is unspecified_argument:
			dataset = A.pull('dataset', None, ref=True)
		
		super().__init__(A, **kwargs)
		
		self.dataset = dataset
		
		
	# def encode(self, x):
	# 	return self.model.encode(x)
	
		

@fig.Component('inference')
class Inference_Task(Task):
	def __init__(self, A, inference=unspecified_argument, solver_types=unspecified_argument, **kwargs):
		if inference is unspecified_argument:
			inference = A.pull('inference', None)
		
		if solver_types is unspecified_argument:
			solver_types = A.pull('solver-types', [])
		
		super().__init__(A, **kwargs)
		
		self.inference = inference
		self.solver_types = solver_types
		self.solvers = None
		
		self._scores = []
		self._results = []
		self._A = A
		
	
	def build_model_from_labels(self, labels):
	
		A = self._A
		
		ltypes = []
		solvers = []
		
		for lbl in labels.t():
			
			if lbl.float() == lbl.int(): # classification
				ltypes.append('classifier')
				classifer = A.pull('classifier', None)
				if classifer is None:
					classifer = GradientBoostingClassifier()
				solvers.append(classifer)
			else: # regression
				ltypes.append('regressor')
				regressor = A.pull('regressor', None)
				if regressor is None:
					regressor = GradientBoostingRegressor()
				solvers.append(regressor)
			
		self._label_types = ltypes
		
		scores = []
		results = []
		
		if 'classifier' in ltypes:
			scores.extend(['f1-score', 'precision', 'recall', 'accuracy'])
			results.extend(['confusion-matrix'])
		
		if 'regressor' in ltypes:
			scores.append(['mse-loss'])
		
		return solvers
		
	
	def fit(self, data, targets=None):
		assert targets is not None, 'No labels provided'
		
		if self.solvers is None:
			self.solvers = self.build_model_from_labels(targets)

		for solver, lbl in zip(self.solvers, targets.t().unsqueeze(-1)):
			solver.fit(data, lbl)
			
		return self
	
	
	def predict(self, data):
		assert self.solvers is not None
		
		return torch.stack([solver.predict(data).view(-1) for solver in self.solvers])
	
	
	def get_scores(self):
		return self._scores
	
	
	def get_results(self):
		return self._results
	
	
	def _compute(self, info=None):
		
		if self.solvers is None:
		
			if self.dataset is None:
				self.dataset = info.get_dataset()
		
			if self.solvers is None:
				self.build_model_from_dataset(self.dataset)
		
			samples = self.dataset.get_observations()
			labels = self.dataset.get_labels()
			if len(labels.shape) == 1:
				labels = labels.unsqueeze(-1)
				
			self.fit(samples, labels)
			
		
		
		
		
		pass
		
		