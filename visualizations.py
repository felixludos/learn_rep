# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
import torch
from torch.utils.data import TensorDataset
# %matplotlib tk
#plt.switch_backend('Qt5Agg') #('Qt5Agg')
from foundation import util
from foundation.old import train as trn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import animation



def viz_latent(Q, figax=None, figsize=(9, 3), lim_y=None):
	Xs = np.arange(Q.shape[-1]) + 1
	inds = np.stack([Xs] * Q.shape[0])

	vals = Q.cpu().numpy()
	df = pd.DataFrame({'x': inds.reshape(-1), 'y': vals.reshape(-1)})

	if figax is None:
		figax = plt.subplots(figsize=figsize)
	fig, ax = figax

	# plt.figure(fig.num)
	plt.sca(ax)

	hue = None
	split = False
	color = 'C0'
	inner = 'box'
	palette = None

	sns.violinplot(x='x', y='y', hue=hue,
				   data=df, split=split, color=color, palette=palette,
				   scale="count", inner=inner, gridsize=100, )
	if lim_y is not None:
		plt.ylim(-lim_y, lim_y)
	# plt.title('Distributions of Latent Dimensions')
	# plt.xlabel('Dimension')
	# plt.ylabel('Values')
	# plt.tight_layout()
	# border, between = 0.02, 0.01
	# 	plt.subplots_adjust(wspace=between, hspace=between,
	# 						left=border, right=1 - border, bottom=border, top=1 - border)
	return fig, ax

def get_traversal_vecs(Q, steps=32, bounds=None, mnmx=None):

	N, D = Q.shape
	S = steps
	#
	# dH, dW = util.calc_tiling(D)
	#
	# # bounds = (-2,2)
	#
	# save_inds = [0, 1, 2, 3]
	#
	# saved_walks = []

	I = torch.eye(D, device=Q.device).view(1, 1, D, D)

	deltas = torch.linspace(0, 1, steps=S, device=Q.device)
	deltas = torch.stack([deltas] * D)  # DxS

	if mnmx is None:
		if bounds is None:
			mnmx = (Q.min(0)[0].view(D, 1), Q.max(0)[0].view(D, 1))
		else:
			mnmx = torch.ones(D,device=Q.device)*bounds[0], torch.ones(D,device=Q.device)*bounds[1]

	mn, mx = mnmx
	mn, mx = mn.view(D, 1), mx.view(D, 1)

	deltas *= mx - mn
	deltas += mn
	deltas = deltas.t().unsqueeze(0).expand(N, S, D).unsqueeze(-1)

	Q = Q.unsqueeze(1).unsqueeze(-1).expand(N, S, D, D)

	vecs = Q * (1 - I) + deltas * I
	vecs = vecs.permute(0, 3, 1, 2) # NxDxSxD (batch, which dim, steps, vec)

	return vecs

def get_traversals(vecs, model, pbar=None): # last dim must be latent dim (model input)

	*shape, D = vecs.shape

	dataset = TensorDataset(vecs.view(-1,D))

	loader = trn.get_loaders(dataset, batch_size=64, shuffle=False)

	if pbar is not None:
		loader = pbar(loader)

	imgs = []

	for Q, in loader:
		with torch.no_grad():
			imgs.append(model(Q))

	imgs = torch.cat(imgs)

	_, *img_shape = imgs.shape

	return imgs.view(*shape,*img_shape)

def tile_dims(full):
	tH, tW = util.calc_tiling(len(full))
	N, S, C, H, W = full.shape
	full = full.view(tH, tW, S, C, H, W)
	full = full.permute(2, 3, 0, 4, 1, 5).contiguous().view(S, C, tH * H, tW * W)
	return full

def get_traversal_anim(frames, vals=None, text_fmt='{:2.2f}', text_size=12, scale=1, fps=20):

	frames = frames.permute(0,2,3,1).cpu().numpy()
	if vals is not None:
		vals = vals.cpu().numpy()

	H, W, C = frames[0].shape
	asp = W/H
	fig = plt.figure(figsize=(asp, 1), dpi=int(H*scale),)

	ax = plt.axes([0, 0, 1, 1], frameon=False)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	plt.autoscale(tight=True)

	im = plt.imshow(frames[0])
	# plt.axis('off')
	# plt.tight_layout()
	if vals is not None:
		txt = plt.text(5,text_size*H//64, text_fmt.format(vals[0]), size=text_size)
	pass

	plt.close()

	def init():
		im.set_data(frames[0])
		if vals is not None:
			txt.set_text(text_fmt.format(vals[0]))

	def animate(i):
		im.set_data(frames[i])
		if vals is not None:
			txt.set_text(text_fmt.format(vals[i]))
		return im

	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames), interval=1000//fps)
	return anim

def compute_diffs(walks, dsteps=10):
	'''
	computes the Linf distance in pixel space between an image and the image where
	a single latent dimension is perturbed by approximately half of the range
	of that latent dim
	'''

	B, D, S, C, H, W = walks.shape

	diffs = (walks[:, :, dsteps:] - walks[:, :, :-dsteps]).abs().max(-3)[0]
	diffs = diffs.view(B, D, (S - dsteps) * H * W).mean(-1)
	return diffs


def viz_interventions(dists, figax=None, figsize=(9, 3), color='C2'):

	vals = dists.cpu().numpy()
	Xs = np.arange(vals.shape[-1]) + 1
	inds = np.stack([Xs] * vals.shape[0])
	df = pd.DataFrame({'x': inds.reshape(-1), 'y': vals.reshape(-1)})
	# df['moment']='log(sigma)'

	hue = None
	split = False
	# color = 'C2'
	inner = 'box'
	palette = None

	if figax is None:
		figax = plt.subplots(figsize=figsize)
	fig, ax = figax
	plt.sca(ax)
	sns.violinplot(x='x', y='y', hue=hue,
				   data=df, split=split, color=color, palette=palette,
				   scale="count", inner=inner, gridsize=100, cut=0)
	# plt.title('Intervention Effect on Image')
	# plt.xlabel('Dimension')
	# plt.ylabel('Effect')
	# plt.tight_layout()


	return fig, ax





