
import omnifig as fig

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import distributions as distrib

from foundation import util
from foundation.data import Intervention_Sampler


def compute_response(Q, encode, decode, include_q2=False,
                     force_different=False, skip_shuffle=False):
	N, D = Q.size()
	
	
	hyb = []
	resp = []
	
	q2s = None
	Q2 = [] if include_q2 else None
	
	if force_different and not skip_shuffle:
		Q = Q[torch.randperm(len(Q))]
	
	for idx in range(D):

		V = Q[:, idx]
		if force_different:
			U = V.clone()
			U[:-1] = V[1:]
			U[-1] = V[0]
		else:
			U = V[torch.randperm(len(V))]
		loader = DataLoader(TensorDataset(Q, U), batch_size=100, drop_last=False)
		
		H, Y = [], []
		
		for q, u in loader:
			
			with torch.no_grad():
				h = q.clone()
				h[:, idx] = u
				
				H.append(h)
				y = encode(decode(h))
				if isinstance(y, distrib.Distribution):
					y = y.loc
				Y.append(y)
				
				if Q2 is not None:
					q2 = encode(decode(q))
					if isinstance(q2, distrib.Distribution):
						q2 = q2.loc
					Q2.append(q2)
			
		hyb.append(torch.cat(H))
		resp.append(torch.cat(Y))
		
		if Q2 is not None:
			q2s = torch.cat(Q2)
			Q2 = None
	
	out = [torch.stack(hyb), torch.stack(resp)]
	if q2s is not None:
		out.append(q2s)
	
	return out


def response_mat(Q, encode, decode, scales=None, dist_type='rms', **resp_kwargs):
	
	H, Y, Q2 = compute_response(Q, encode, decode, include_q2=True, **resp_kwargs)
	
	R = Y - Q2.unsqueeze(0)
	
	if scales is not None:
		R /= scales.view(1, 1, -1)
	
	if dist_type == 'rms':
		R = R.pow(2).mean(1).sqrt()
	elif dist_type == 'sqr':
		R = R.pow(2).mean(1)
	elif dist_type == 'abs':
		R = R.abs().mean(1)
	elif dist_type == 'l1':
		R = R.abs().sum(1)
	elif dist_type == 'l2':
		R = R.pow(2).sum(1).sqrt()
	
	return R


# from full interventions

def sample_full_interventions(dataset, num_groups=50, device='cuda', pbar=None):
	
	sampler = Intervention_Sampler(dataset)
	
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


def factor_reponses(encode, decode, factor_samples, resp_kwargs={}, include_q=False,
                    pbar=None, factor_names=None):

	Fs = []
	allQs = [] if include_q else None
	
	for i, groups in enumerate(factor_samples):
		
		N, G, C, H, W = groups.size()
		
		with torch.no_grad():
			
			Q = encode(groups.view(N*G, C, H, W))
			if isinstance(Q, distrib.Distribution):
				Q = Q.loc
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
				Ms.append(response_mat(q, encode, decode, **resp_kwargs))
		
		Fs.append(torch.stack(Ms))
	
	out = [torch.stack(Fs)]
	if include_q:
		out.append(allQs)
	return out


@fig.Script('evaluate-responses')
def eval_responses(A, run=None):
	
	if run is None:
		run = A.pull('run')
		
	



