


import sys, os, time
import shutil
import traceback
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.multiprocessing as mp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

import foundation as fd
from foundation import data as datautils
from foundation import models
from foundation import util
from foundation import train as trn


def get_estimator(A):
	raise NotImplementedError

def main(A):

	if 'seed' not in A:
		A.seed = util.gen_random_seed()
	seed = A.pull('seed')
	util.set_seed(seed)

	assert 'dataset' in A, 'No dataset provided'
	info = A.dataset

	if 'train' in info:
		A.mode = 'train' if A.train else 'test'
	mode = A.dataset.pull('mode', 'train') # 'mode' overrides 'train'

	dataset = trn.default_load_data(info, mode=mode)
	assert isinstance(dataset, datautils.Device_Dataset), 'Dataset must be in memory'

	idx = A.pull('buffer_idx', 0)

	buffer = getattr(dataset, dataset._buffers[idx])

	N = A.pull('N', len(buffer))

	assert N <= len(dataset), f'Not enough samples: {N} vs {len(dataset)}'

	order = slice(N)
	if A.pull('shuffle', False):
		util.set_seed(seed)
		order = torch.randperm(len(dataset))[order]

	print(f'Loaded a dataset size: {len(dataset)}')

	samples = buffer[order]

	print(f'Using {len(samples)}')


	alg = A.pull('alg', 'umap')

	if alg == 'umap':

		pass
	elif alg == 'pca':
		pass
	else:
		raise Exception(f'unknown alg: {alg}')

	pass

if __name__ == '__main__':
	sys.exit(trn.main(argv=sys.argv, cmd=main))




