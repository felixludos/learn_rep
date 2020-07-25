#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
#%matplotlib tk
from matplotlib import cm

import foundation as fd
from foundation import util


@fd.Component('point-enc')
class PointEncoder(fd.Encodable, fd.Visualizable, fd.Schedulable, fd.Model):

	def __init__(self, A):
		in_shape = A.pull('in_shape', '<>din')
		latent_dim = A.pull('latent_dim', '<>dout')

		assert 'transform' in A, 'transofrm required to go from image to point cloud'
		A.transform.in_shape = in_shape

		transform = A.pull('transform') # converts the image into a point cloud

		assert 'pointnet' in A, 'pointnet required for this encoder'
		pout, N = transform.dout

		A.pointnet.pin = pout
		A.pointnet.n_points = N

		pointnet = A.pull('pointnet')

		super().__init__(in_shape, latent_dim)

		self.transform = transform
		self.pointnet = pointnet

		# self.set_optim(A)
		# self.set_scheduler(A)

	def _visualize(self, out, logger): # TODO

		if self._viz_counter % 2 == 0:
			for m in self.pointnet.tfms:
				if isinstance(m, fd.Visualizable):
					m.visualize(out, logger)

		pass

	def forward(self, x): # images

		p = self.transform(x)
		q = self.pointnet(p)

		return q


@fd.AutoComponent('patch-points')
class Patch_Points(fd.Model):
	'''
	Converts an image into a set of unordered points where each point is a concat of the pixels of
	a unique patch of the image.
	'''
	def __init__(self, in_shape, include_coords=False,
	             patch_size=4, stride=2, dilation=1, padding=0):

		C, H, W = in_shape

		try:
			len(patch_size)
		except TypeError:
			patch_size = (patch_size, patch_size)

		try:
			len(stride)
		except TypeError:
			stride = (stride, stride)

		try:
			len(dilation)
		except TypeError:
			dilation = (dilation, dilation)

		try:
			len(padding)
		except TypeError:
			padding = (padding, padding)

		pout = C*patch_size[0]*patch_size[1]

		Nh = (H + 2 * padding[0] - dilation[0] * (patch_size[0] - 1) - 1) // stride[0] + 1
		Nw = (W + 2 * padding[1] - dilation[1] * (patch_size[1] - 1) - 1) // stride[1] + 1
		N = Nh * Nw

		if include_coords:
			pout += 2

		super().__init__(in_shape, (pout, N))

		self.pout = pout

		self.transform = nn.Unfold(kernel_size=patch_size, stride=stride,
		                           dilation=dilation, padding=padding)

		if include_coords:
			self.register_buffer('coords', torch.from_numpy(np.mgrid[:1:1j*Nh,:1:1j*Nw]).view(2, -1).float())
		else:
			self.coords = None

	def forward(self, x):
		p = self.transform(x)
		if self.coords is not None:
			B = p.size(0)
			p = torch.cat([p, self.coords.expand(B,*self.coords.size())],1)
		return p



@fd.AutoComponent('1dconv-net')
def make_pointnet(pin, pout, hidden=[],
                  nonlin='prelu', output_nonlin=None):
	nonlins = [nonlin] * len(hidden) + [output_nonlin]
	hidden = [pin] + list(hidden) + [pout]

	layers = []
	for in_dim, out_dim, nonlin in zip(hidden, hidden[1:], nonlins):
		layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1))
		if nonlin is not None:
			layers.append(util.get_nonlinearity(nonlin))

	net = nn.Sequential(*layers)
	net.pin = pin
	net.pout = pout
	return net


@fd.Component('point-net')
class PointNet(fd.Trainable_Model):
	def __init__(self, A):

		pin = A.pull('pin') # should be the number of channels of the points
		dout = A.pull('latent_dim', '<>dout')

		n_points = A.pull('n_points', None)

		create_module = A.pull('modules')

		super().__init__(din=pin, dout=dout)

		modules = []

		nxt = create_module.current()
		assert nxt is not None, 'no point-net modules provided'
		nxt.pin = pin
		nxt.N = n_points

		for module in create_module:

			if hasattr(module, 'nout'):
				n_points = module.nout

			modules.append(module)

			nxt = create_module.current()
			if nxt is not None:
				nxt.pin = module.pout
				nxt.pin1 = module.pout1
				nxt.pin2 = module.pout2
				nxt.N = n_points
				if nxt._type == 'point-dual':
					if 'left' in nxt:
						nxt.left.pin = module.pout1
					if 'right' in nxt:
						nxt.right.pin = module.pout2

		pout = module.pout
		assert pout is not None, f'last module must have a single output: {module}'

		self.tfms = nn.Sequential(*modules)

		if 'pool' in A and '_type' in A.pool:
			A.pool.din = (pout, n_points)
			# A.pool.pin = pout
			# A.pool.N = n_points
		self.pool = A.pull('pool', None)
		if self.pool is not None:
			pout = self.pool.dout

		if 'final' in A:
			A.final.din = pout
			A.final.dout = dout
		final = A.pull('final', None)

		if final is not None: # make sure final will produce the right output
			assert final.din == pout and final.dout == dout, f'invalid {final.din} v {pout} and {final.dout} v {dout}'
		elif pout != dout: # by default fix output to correctly sized output using a dense layer
			final = nn.Linear(pout, dout)
		self.final = final

	def forward(self, pts):

		pts = self.tfms(pts)

		B, _, *rest = pts.size()
		if len(rest):
			if rest[0] > 1:
				pts = self.pool(pts)
			pts = pts.view(B,-1)
		out = pts

		if self.final is not None:
			out = self.final(out)

		return out

# Abstract modules

class PointNetModule(fd.Model):
	def __init__(self, pin=None, pout=None,
	             pin1=None, pout1=None, pin2=None, pout2=None):
		super().__init__(pin, pout)
		self.pin = pin
		self.pout = pout

		self.pin1 = pin1
		self.pout1 = pout1
		self.pin2 = pin2
		self.pout2 = pout2


class PointSplit(PointNetModule):
	def __init__(self, pin, pout1, pout2):
		super().__init__(pin=pin, pout1=pout1, pout2=pout2)

class PointTransform(PointNetModule):
	def __init__(self, pin, pout):
		super().__init__(pin=pin, pout=pout)

class PointParallel(PointNetModule):
	def __init__(self, pin1, pin2, pout1, pout2):
		super().__init__(pin1=pin1, pin2=pin2, pout1=pout1, pout2=pout2)

	def __call__(self, pts): # so the point net can connect modules by nn.Sequential
		return super().__call__(*pts)

class PointJoin(PointNetModule):
	def __init__(self, pin1, pin2, pout):
		super().__init__(pin1=pin1, pin2=pin2, pout=pout)

	def __call__(self, pts): # so the point net can connect modules by nn.Sequential
		return super().__call__(*pts)


# Point-net operations


@fd.AutoComponent('point-split')
class PointSplitter(PointSplit):
	def __init__(self, pin, split):
		super().__init__(pin=pin, pout1=split, pout2=pin-split)

		assert pin > split, f'not enough dimensions to split: {pin} vs {split}'
		self.split = split

	def extra_repr(self):
		return f'split={self.split}'

	def forward(self, p):
		return p[:,:self.split], p[:,self.split:]

@fd.AutoComponent('point-buffer')
class PointBuffer(PointSplit):
	def __init__(self, pin, channels, N):
		super().__init__(pin=pin, pout1=channels, pout2=pin)

		self.buffer = nn.Parameter(torch.randn(channels, N), requires_grad=True)

	def extra_repr(self):
		return 'chn={}, N={}'.format(*self.buffer.size())

	def forward(self, p):
		B = p.size(0)

		p1 = self.buffer.expand(B, *self.buffer.shape)
		p2 = p

		return p1, p2



@fd.AutoComponent('point-transform-net')
class PointTransformNet(PointTransform):
	def __init__(self, net):
		super().__init__(net.pin, net.pout)

		self.net = net

	def forward(self, p):
		return self.net(p)

@fd.AutoComponent('point-self')
class PointSelfTransform(PointTransformNet):
	def __init__(self, pin, pout, hidden=[], nonlin='prelu', output_nonlin=None):
		super().__init__(make_pointnet(pin, pout, hidden=hidden, nonlin=nonlin,
		                         output_nonlin=output_nonlin))



@fd.AutoComponent('point-dual')
class PointDualTransform(PointParallel):
	def __init__(self, left=None, right=None, pin1=None, pin2=None):

		assert left is not None or pin1 is not None
		assert right is not None or pin2 is not None

		if left is not None:
			pin1 = left.pin
			pout1 = left.pout
		else:
			pout1 = pin1

		if right is not None:
			pin2 = right.pin
			pout2 = right.pout
		else:
			pout2 = pin2

		super().__init__(pin1, pin2, pout1, pout2)

		self.left = left
		self.right = right

	def forward(self, p1, p2):
		if self.left is not None:
			p1 = self.left(p1)
		if self.right is not None:
			p2 = self.right(p2)
		return p1, p2

@fd.AutoComponent('point-swap')
class PointSwap(PointParallel):
	def __init__(self, pin1, pin2):
		super().__init__(pin1=pin1, pin2=pin2, pout1=pin2, pout2=pin1)

	def forward(self, p1, p2):
		return p2, p1



@fd.AutoComponent('point-cat')
class PointJoiner(PointJoin):
	def __init__(self, pin1, pin2):
		super().__init__(pin1=pin1, pin2=pin2, pout=pin1+pin2)

	def forward(self, p1, p2): # TODO make sure shapes work out
		return torch.cat([p1, p2], dim=1)

@fd.AutoComponent('point-wsum')
class PointWeightedSum(fd.Cacheable, fd.Visualizable, PointJoin):

	def __init__(self, pin1, pin2, heads=1, keys=1, norm_heads=False, sum_heads=True,
	             gumbel=None, gumbel_min=0.1, gumbel_delta=2e-4):
		super().__init__(pin1=pin1, pin2=pin2, pout=pin1)
		self.nout = heads if sum_heads else heads*keys

		self.weights = nn.Conv1d(pin2, heads*keys, kernel_size=1)

		self.norm_heads = norm_heads
		self.sum_heads = sum_heads

		self.keys = keys
		self.heads = heads
		self.N_out = self.keys * self.heads
		
		if gumbel is not None and gumbel <= 0:
			gumbel = None
		self.gumbel = gumbel
		self.gumbel_delta = gumbel_delta
		self.gumbel_start = gumbel
		self.gumbel_min = gumbel_min

		self.register_cache('_w')
		self.cmap = cm.get_cmap('seismic')

	def compute_weights(self, p): # optionally use gumbel softmax
		return self.weights(p)

	def extra_repr(self):
		return f'heads={self.heads}, keys={self.keys}'

	def _visualize(self, out, logger):

		if self._w is not None:
			w = self._w
			
			brightness = 100*10**(self.gumbel is None)

			B, G, K, N = w.shape

			H, W = util.calc_tiling(N)

			g = w[0]
			g = g.view(G, K, H, W)
			out.key_selections = g
			g = g.view(G*K, H, W)

			g = torchvision.utils.make_grid(g.unsqueeze(1).mul(brightness).clamp(max=1), nrow=K, padding=1, pad_value=1)[:1]#.mean(dim=0).unsqueeze(0)

			# g = torch.from_numpy(self.cmap(g.numpy())).permute(0, 3, 1, 2)[:,:3]
			# g = torchvision.utils.make_grid(g, nrow=K, padding=1, pad_value=1)#.norm(p=1, dim=0)

			logger.add('image', 'patches-1', g)
			
			g = w.sum(0) / B
			g = g.view(G, K, H, W)
			out.key_selections = g
			g = g.view(G * K, H, W)
			
			g = torchvision.utils.make_grid(g.unsqueeze(1).mul(brightness).clamp(max=1), nrow=K, padding=1, pad_value=1)[:1]  # .mean(dim=0).unsqueeze(0)
			
			# g = torch.from_numpy(self.cmap(g.numpy())).permute(0, 3, 1, 2)[:,:3]
			# g = torchvision.utils.make_grid(g, nrow=K, padding=1, pad_value=1)#.norm(p=1, dim=0)
			
			logger.add('image', 'patches-avg', g)

	def forward(self, p1, p2): # p1 (B, C, N)
		w = self.compute_weights(p2) # (B, GK, N)
		B, GK, N = w.shape
		G = self.heads
		K = self.keys

		if self.gumbel is not None:
			if self.training:
				self.gumbel = max(self.gumbel - self.gumbel_delta, self.gumbel_min)
			w += torch.rand_like(w).log().mul(-1).log().mul(-1)
			w /= self.gumbel

		w = w.view(B, G, K*N) if self.norm_heads else w.view(B, G, K, N)
		w = F.softmax(w, dim=-1).view(B, G*K, N)

		self._w = w.view(B,G,K,N).cpu().detach()

		w = w.permute(0,2,1)
		v = p1 @ w

		if self.sum_heads:
			C = p1.size(1)
			v = v.view(B, C, G, K).sum(-1)

		return v # (B, C, G*K)

# @fd.AutoComponent('point-wsum')
# class PointWeightedSum(fd.Cacheable, fd.Visualizable, PointJoin):
#
# 	pass


@fd.AutoComponent('pool-points')
class Pool_Points(PointTransform):
	def __init__(self, pin, fn='max', p=2):
		super().__init__(pin, pin)

		assert fn in {'max', 'avg', 'sum', 'std', 'norm'}, f'unknown pool type: {fn}'

		self.fn = fn
		self.p = p

	def extra_repr(self):
		return f'{self.fn}'

	def forward(self, p):

		if self.fn == 'max':
			return p.max(-1, keepdim=True)[0]
		if self.fn == 'avg':
			return p.mean(-1, keepdim=True)
		if self.fn == 'sum':
			return p.sum(-1, keepdim=True)
		if self.fn == 'std':
			return p.std(-1, keepdim=True)
		if self.fn == 'norm':
			return p.norm(p=self.p, dim=-1, keepdim=True)

@fd.AutoComponent('concat-points')
class Concat_Points(PointTransform):
	def __init__(self, pin, N=None, groups=1):
		if N is None:
			print('WARNING: no number of points set')
			N = 1

		super().__init__(pin, N*pin//groups)

		self.groups = groups

	def forward(self, p):
		B = p.size(0)
		return p.permute(0,2,1).contiguous().view(B, self.groups, -1)




