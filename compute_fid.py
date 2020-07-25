import sys  #, traceback, ipdb
from tqdm import tqdm

import foundation as fd
from foundation import util
from foundation.old import train as trn

_script_name = 'fid'

@trn.Script(_script_name)
def compute_fid(dataset=None, model=None,
                
                dataset_args={},
                
                n_samples=50000, batch_size=50,
                fid_dim=2048, device='cuda',
                save_name=None, overwrite=False, pbar=False):
	
	pbar = tqdm if pbar else None
	
	print('Loading inception model...', end='')
	inception_model = fd.eval.fid.load_inception_model(dim=fid_dim, device=device)
	print('done')
	
	assert dataset is not None or model is not None, 'nothing to compute'
	
	if dataset is not None:
		print('Loading dataset')
		
		dataset = trn.get_dataset(dataset, **dataset_args)
	
		loader = trn.get_loaders(dataset, batch_size=batch_size)
		true_loader = util.make_infinite(loader)
		def true_fn(N):
			return util.to(true_loader.demand(N), device)[0]
		
		print('Computing dataset (gt) fid stats')
		
		ds_stats = fd.eval.fid.compute_inception_stat(true_fn, inception=inception_model,
		                                              batch_size=batch_size, n_samples=n_samples,
		                                              pbar=pbar)
		
		print('Dataset (gt) fid stats computed.')
		
		if save_name is not None:
			pass
		
	
	def make_gen_fn():
		loader = util.make_infinite(valloader)
		
		def gen_fn(N):
			# assert N == batch_size, '{} vs {}'.format(N, batch_size)
			x = util.to(loader.demand(N), device)[0]
			return self(x)
		
		return gen_fn
	
	m, s = fd.eval.fid.compute_inception_stat(make_gen_fn(), inception=inception_model,
	                                          batch_size=batch_size, n_samples=n_samples,
	                                          pbar=tqdm if 'inline' in A and A.inline else None)
	
	
	pass


if __name__ == '__main__':
	argv = sys.argv
	argv = [_script_name] + argv[1:]
	trn.main_script(argv)
