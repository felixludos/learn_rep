
import sys, os, time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['FOUNDATION_RUN_MODE'] = 'jupyter'
# os.environ['FOUNDATION_SAVE_DIR'] = '/is/ei/fleeb/workspace/chome/trained_nets'
# os.environ['FOUNDATION_DATA_DIR'] = '/is/ei/fleeb/workspace/local_data'
# %load_ext autoreload
# %autoreload 2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as O
import torch.distributions as distrib
import torch.multiprocessing as mp
import torchvision.models
import torchvision
from torch.utils.data import Dataset, DataLoader
import gym
from io import BytesIO
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import inspect
import numpy as np
import h5py as hf
# %matplotlib notebook
# %matplotlib tk
import matplotlib.pyplot as plt
import seaborn as sns
#plt.switch_backend('Qt5Agg') #('Qt5Agg')
import foundation as fd
from foundation import models
from foundation import util
from foundation import train
from foundation import sim as SIM
#from foundation.util import replicate, Cloner
from scipy import stats

def print_info(f):
	print(list(f.keys()), list(f.attrs.keys()))
	for k in f.keys():
		print('{}: {} {}'.format(k, f[k].dtype, f[k].shape))

def main(argv=None):
	
	if argv is None:
		argv = sys.argv
	
	print(argv)
	
	cat = argv[-1]
	
	dataroot = os.environ['FOUNDATION_DATA_DIR']
	dataset_name = 'mpi3d'
	
	N = 1036800
	
	indices = torch.arange(N).long()
	
	# cat = 'real'
	
	src_name = f'{dataset_name}_{cat}.npz'
	
	path = os.path.join(dataroot, dataset_name, src_name)
	
	print(f'{cat} {dataset_name} {path}')
	
	data = np.load(path)
	
	images = data['images']

	print(f'Data loaded: {images.shape}')
	
	target_path = f'{dataset_name}_{cat}_full.h5'
	
	print('Dest: {}'.format(target_path))
	print('Name: {}, Size: {}'.format(os.path.basename(target_path), len(indices)))
	with hf.File(target_path, 'w') as tgt:
		tgt.create_dataset('indices', data=indices)
		tgt.create_dataset('images', data=images)
		print_info(tgt)
	
	print('done')
	
	pass

if __name__ == '__main__':
	main()




