
import omnifig as fig

import torch

from foundation.op.datasets import MPI3D, Shapes3D, CelebA
from foundation.data import Dataset, JointFactorSampler

@Dataset('full-mpi3d')
class Full_MPI3D(MPI3D):
	
	def __init__(self, A, sort=None, **kwargs):
		
		if sort is None:
			sort = A.pull('sort', True)
		
		super().__init__(A, mode='train', **kwargs)
		
		img1, idx1 = self.images, self.indices
		
		with A.silenced():
			super().__init__(A, mode='test', **kwargs)
		
		self.images = torch.cat([img1, self.images])
		self.indices = torch.cat([idx1, self.indices])
	
		self.order = self.indices.argsort() if sort else None
	
	def __getitem__(self, item):
		if self.order is not None:
			item = self.order[item]
		return super().__getitem__(item)
		

