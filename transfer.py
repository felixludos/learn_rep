

import sys, os, time, shutil#, traceback, ipdb
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import configargparse

import numpy as np
#%matplotlib tk
import matplotlib.pyplot as plt


import foundation as fd
from foundation import models
from foundation import util
from foundation import train as trn
from foundation import data as datautils


@trn.Modification('trans')
class Multi_Dataset(datautils.Info_Dataset, datautils.Device_Dataset, datautils.Batchable_Dataset):
	'''
	
	dataset.get_factor_order() should returns a ordered list of factors
	dataset.get_factor_sizes() should return a dict(name -> len)
	dataset.get_labels() should return full labels tensor
	
	'''
	def __init__(self, dataset, A=None, folds=None, mix=None, limits=None):
		
		if A is not None:
			folds = A.pull('folds')
			default_fold = A.pull('default_fold', 'train')
			fold_by = A.pull('fold_by', 'index')
			
			mix = A.pull('mix', {default_fold:1})
			
			limits = A.pull('limits', None)
		
		super().__init__(dataset.din, dataset.dout)
		
		print('Found {} folds: {}'.format(len(folds), ', '.join(folds.keys())))
		
		if A is not None:
			if fold_by == 'index':
				folds = {name:{int(k):v for k,v in fold.items()} for name, fold in folds.items()}
			else:
				inds = {name:i for i, name in enumerate(dataset.get_factor_order())}
				folds = {name:{inds[k]:v for k,v in fold.items()} for name, fold in folds.items()}
			folds = self._compute_sels(dataset.get_labels(), folds)
		self.folds = folds
		
		self.fid_stats = None
		
		self.limits = limits
		self._limited_folds = {name:(inds[torch.randperm(len(inds))[:limits[name]]] if name in limits else inds)
		                       for name,inds in folds.items()} if limits is not None else folds
		
		self.register_buffer('dataset', dataset)

		self.default_mix = mix
		self.register_buffer('inds', None)
		
		self.set_mix(**mix)

	def split(self, info):
		val_per = info.pull('val_split', 0.1)
		shuffle_split = info.pull('shuffle_split', True)
		
		assert 0 < val_per < 1, f'invalid: {val_per}'
		
		train = {}
		val = {}
		
		for name, inds in self.folds.items():
			N = max(int(len(inds)*val_per),1)
			if shuffle_split:
				sel = torch.randperm(len(inds), device=inds.device)
				train[name] = inds[sel[:-N]]
				val[name] = inds[sel[-N:]]
			else:
				train[name] = inds[:-N]
				val[name] = inds[-N:]

		return Multi_Dataset(self.dataset, folds=train, mix=self.mix, limits=self.limits), \
		       Multi_Dataset(self.dataset, folds=val, mix=self.mix, limits=None) # validation set has not limits

	def _compute_sels(self, lbls, folds):
		
		sels = {}
		
		for name, fold in folds.items():
			sel = torch.ones(len(lbls), dtype=bool)
			
			for factor, vals in fold.items():
				sel *= sum(lbls[:,factor] == v for v in vals).bool()
			
			sels[name] = torch.arange(len(lbls))[sel]#[torch.logical_not(sel)]
			if len(sels[name]) == 0:
				print(f'WARNING: Fold "{name}" has 0 samples')
		
		return sels
		
	def pre_epoch(self, mode, epoch):
		self.resample_mix()
		self.dataset.pre_epoch(mode, epoch)

	def resample_mix(self): # remember to move to self.device
		
		folds = self._limited_folds
		
		if len(self.mix) == 1:
			self.inds = folds[next(iter(self.mix))].to(self.device)
			
		inds = []
		
		for name, N in self.mix.items():
			
			L = len(folds[name])
			
			if N < L:
				inds.append(folds[name][torch.randperm(L)[:N]])
			else:
				rem = L % N
				mul = L // N
				inds.extend(mul*[folds[name]])
				if rem > 0:
					inds.append(folds[name][torch.randperm(L)[:rem]])
		
		self.inds = torch.cat(inds).to(self.device)

	def __len__(self):
		return len(self.inds)

	def __getitem__(self, idx):
		return self.dataset[self.inds[idx]]

	def set_mix(self, **mix):
		assert len(mix)
		
		folds = self._limited_folds # use limited
		
		if len(mix) == 1:
			mix = {k:len(folds[k]) for k in mix}
		else:
			atom = min(mix.values())
			base, prop = min(mix.items(), key=lambda x: len(folds[x[0]]) / x[1])
			weight = len(folds[base]) / prop
			
			mix = {k: max(int(weight * v), 1) for k, v in mix.items()}
		
		self.mix = mix
		self.absolute = False
		self.resample_mix()
		print('Set mix: {}'.format(self.mix))
		return
	
	def set_full(self, *names):
		
		folds = self.folds
		
		inds = []
		mix = {}
		
		for name in names:
			assert name in folds, f'not found {name}'
			mix[name] = len(folds[name])
			inds.append(folds[name])
		
		self.inds = torch.cat(inds)
		self.mix = mix
		
		self.absolute = True
		print('Fold set to: {}'.format(', '.join(names)))





# old


class Mechanism_Transfer(datautils.Subset_Dataset):
	def __init__(self, dataset, info):
		
		setting = info.pull('setting', 'train')
		# assert setting in {'train', 'test', 'update'}
		
		mechanisms = info.pull('mechanisms', {})
		if 'train' not in mechanisms:
			mechanisms['train'] = dataset.factor_order.copy()
		if 'update' not in mechanisms:
			mechanisms['update'] = mechanisms['train'].copy()
		if 'eval' not in mechanisms:
			mechanisms['eval'] = mechanisms['update'].copy()
		print('Mechanisms: (current mode: {}): {}'.format(setting, mechanisms[setting]))
		
		default_values = info.pull('default_class', 'middle')
		if not isinstance(default_values, list):
			default_values = [default_values] * len(dataset.factor_order)
		
		super().__init__(dataset)
		
		self.mechanisms = mechanisms
		self.default_views = default_values
		
		self.set_setting(setting)
	
	def set_setting(self, setting):
		assert setting in self.mechanisms, f'{setting} not found'
		
		default_values = iter(self.default_views)
		
		fixed = [(None if m in self.mechanisms[setting] else next(default_values)) for m in self.dataset.factor_order]
		
		sel = self._filter(self.dataset, fixed)
		
		if sel is not None:
			sel = np.arange(len(self.dataset))[sel]
		
		
		self.indices = sel
		try:
			device = self.dataset.get_device()
			if self.indices is not None:
				self.indices = self.indices.to(device)
		except AttributeError:
			pass
		
		self.setting = setting
		print(f'Transfer dataset set to setting: {setting}')
	
	
	def _filter(self, dataset, fixed):
		raise NotImplementedError

@trn.Modification('mtrans-3ds')
class Shapes3D_Mech_Transfer(Mechanism_Transfer):
	
	def _filter(self, dataset, fixed):
		
		lbls = dataset.labels
		
		sel = None
		for i, val in enumerate(fixed):
			if val is not None:
				if val == 'middle':
					vals = sorted(list(set(lbls[:, i].tolist())))
					val = vals[len(vals) // 2]
				valid = lbls[:, i] == val
				if sel is None:
					sel = valid
				else:
					sel *= valid
		
		return sel

@trn.Modification('mtrans-mpi')
class MPI_Mech_Transfer(Mechanism_Transfer):
	
	def _filter(self, dataset, fixed):
		
		inds = dataset.indices
		div = dataset._flr
		
		sel = None
		for d, val in zip(div, fixed):
			if val is not None:
				if val == 'middle':
					val = 0
				valid = inds % d == val
				if sel is None:
					sel = valid
				else:
					sel *= valid
		
		return sel

class Value_Transfer(datautils.Subset_Dataset):
	
	def __init__(self, dataset, info):
		
		nums = info.pull('nums', None)
		fractions = info.pull('fractions', None)
		
		valid = info.pull('valid', None)
		
		sizes = np.array(dataset.factor_sizes)
		
		if nums is None:
			if fractions is not None:
				fractions = np.array(fractions)
				nums = fractions * sizes
				nums = nums.astype(int)
		else:
			nums = np.array(nums)
		
		if nums is not None:
			nums = nums.clip(min=1)
		else:
			nums = [None ] *len(sizes)
		
		if valid is None and nums is None:
			assert False
			valid = [None ] *len(sizes)
		else:
			valid = [torch.arange(n).int() if v is None or v == 'None' else torch.tensor(v).int() for v, n in zip(valid, nums)]
		
		super().__init__(dataset)
		
		sel = self._filter(dataset, valid)
		
		if sel is not None:
			sel = np.arange(len(self.dataset))[sel.numpy().astype(bool)]
		
		self.indices = sel
		try:
			device = self.dataset.get_device()
			if self.indices is not None:
				self.indices = self.indices.to(device)
		except AttributeError:
			pass
	
	def _filter(self, dataset, valid):
		raise NotImplementedError

@trn.Modification('vtrans-3ds')
class Shapes3d_Value_Transfer(Value_Transfer):
	def _filter(self, dataset, valid):
		
		sel = 1
		
		lbls = dataset.labels.clone()
		
		for c, v in zip(lbls.t(), valid):
			v = v.view(1 ,-1)
			c = c.view(-1 ,1)
			s = c.sub(v).eq(0).sum(-1)
			sel *= s
		
		return sel




