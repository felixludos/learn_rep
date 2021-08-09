
import omnifig as fig

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import distributions as distrib

from omnilearn import util




# from full interventions

def sample_full_interventions(sampler, num_groups=50, device='cuda', pbar=None):
	
	D = len(sampler)
	
	factors = []
	
	itr = range(D)
	if pbar is not None:
		itr = pbar(itr, total=D)
		itr.set_description('Sampling interventions')
	else:
		print('Sampling interventions')
	for idx in itr:
		groups = [sampler.full_intervention(idx) for _ in range(num_groups)]
	
		full = torch.stack(groups).to(device)
		
		factors.append(full)
	
	return factors


def response_mat(Q, encode, decode, n_interv=None, scales=None,
                 force_different=False,
                 max_batch_size=None, device=None):
	if scales is not None:
		raise NotImplementedError
	
	@torch.no_grad()
	def response_function(q):
		if device is not None:
			q = q.to(device)
		r = encode(decode(q))
		if isinstance(r, distrib.Normal):
			r = r.loc
		return r
	
	B, D = Q.shape
	
	if n_interv is None:
		n_interv = B
	if max_batch_size is None:
		max_batch_size = n_interv
	
	resps = []
	for i, qi in enumerate(Q.t()):  # Opy(D) (n_interv is parallelized)
		order = torch.randperm(B)
		iorder = order.clone()
		if force_different:
			iorder[1:] = order[:-1]
			iorder[0] = order[-1]
		qsel = slice(0,n_interv) if force_different else torch.randint(B, size=(n_interv,))
		isel = slice(0,n_interv) if force_different else torch.randint(B, size=(n_interv,))
		q = Q[order[qsel]]
		dq = q.clone()
		dq[:, i] = qi[iorder[isel]]
		z = util.process_in_batches(response_function, q, batch_size=max_batch_size)
		dz = util.process_in_batches(response_function, dq, batch_size=max_batch_size)
		resps.append(dz.sub(z).pow(2).mean(0).sqrt())
	return torch.stack(resps)


def conditioned_reponses(encode, decode, factor_samples, resp_kwargs={}, include_q=False,
					pbar=None, factor_names=None):
	'''
	:param encode:
	:param decode:
	:param factor_samples: list with K elements (one for each factor of variation), each element hast
	N sets of full interventions
	:param resp_kwargs:
	:param include_q:
	:param pbar:
	:param factor_names:
	:return:
	'''

	Fs = []
	allQs = [] if include_q else None
	
	def _encode(x):
		q = encode(x)
		if isinstance(q, util.Distribution):
			q = q.bsample()
		return q
	
	for i, groups in enumerate(factor_samples):
		
		N, G, C, *other = groups.size()
		
		with torch.no_grad():
			
			Q = util.process_in_batches(_encode, groups.view(N*G, C, *other), batch_size=64)
			# Q = encode(groups.view(N*G, C, *other))
			# if isinstance(Q, util.Distribution):
			# 	Q = Q.bsample()
			Qs = Q.view(N, G, -1)
			if allQs is not None:
				allQs.append(Qs)
			
			Ms = []
			
			todo = zip(groups, Qs)
			if pbar is not None:
				todo = pbar(todo, total=len(groups))
				if factor_names is not None:
					todo.set_description(factor_names[i])
			
			for group, q in todo:
				Ms.append(response_mat(q, encode, decode, force_different=True, **resp_kwargs).cpu())
		
		Fs.append(torch.stack(Ms))
	
	out = [torch.stack(Fs)]
	if include_q:
		out.append(allQs)
	return out















#
# def compute_response(Q, encode, decode, include_q2=False, mag=None,
#                      force_different=False, skip_shuffle=False):
# 	N, D = Q.size()
#
# 	hyb = []
# 	resp = []
#
# 	q2s = None
# 	Q2 = [] if include_q2 else None
#
# 	if force_different and not skip_shuffle:
# 		Q = Q[torch.randperm(len(Q))]
#
# 	if mag is not None and isinstance(mag, (int, float)):
# 		mag = [mag] * D
#
# 	for idx in range(D):
#
# 		V = Q[:, idx]
# 		if mag is not None:
# 			m = mag[idx]
# 			U = V + m * (-1) ** torch.randint(2, size=(len(V),), device=V.device)
# 		elif force_different:
# 			U = V.clone()
# 			U[:-1] = V[1:]
# 			U[-1] = V[0]
# 		else:
# 			U = V[torch.randperm(len(V))]
# 		loader = DataLoader(TensorDataset(Q, U), batch_size=100, drop_last=False)
#
# 		H, Y = [], []
#
# 		for q, u in loader:
#
# 			with torch.no_grad():
# 				h = q.clone()
# 				h[:, idx] = u
#
# 				H.append(h)
# 				y = encode(decode(h))
# 				if isinstance(y, distrib.Distribution):
# 					y = y.loc
# 				Y.append(y)
#
# 				if Q2 is not None:
# 					q2 = encode(decode(q))
# 					if isinstance(q2, distrib.Distribution):
# 						q2 = q2.loc
# 					Q2.append(q2)
#
# 		hyb.append(torch.cat(H))
# 		resp.append(torch.cat(Y))
#
# 		if Q2 is not None:
# 			q2s = torch.cat(Q2)
# 			Q2 = None
#
# 	out = [torch.stack(hyb), torch.stack(resp)]
# 	if q2s is not None:
# 		out.append(q2s)
#
# 	return out
#
#
# def response_mat(Q, encode, decode, scales=None, dist_type='rms', mag=None, **resp_kwargs):
# 	if isinstance(mag, (float, int)) and scales is not None:
# 		mag = mag * scales
#
# 	H, Y, Q2 = compute_response(Q, encode, decode, include_q2=True, mag=mag, **resp_kwargs)
#
# 	R = Y - Q2.unsqueeze(0)
#
# 	if scales is not None:
# 		R /= scales.view(1, 1, -1)
#
# 	if dist_type == 'rms':
# 		R = R.pow(2).mean(1).sqrt()
# 	elif dist_type == 'sqr':
# 		R = R.pow(2).mean(1)
# 	elif dist_type == 'abs':
# 		R = R.abs().mean(1)
# 	elif dist_type == 'l1':
# 		R = R.abs().sum(1)
# 	elif dist_type == 'l2':
# 		R = R.pow(2).sum(1).sqrt()
#
# 	return R
#



	



