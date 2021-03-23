from omnibelt import set_global_setting
import omnifig as fig
import omnilearn as fd

@fig.Component('sae-run')
class SAE_Run(fd.op.Torch_Run):

	def _gen_name(self, A):

		terms = []
		
		data = A.pull('info.dataset_type', '<>dataset.name', '<>dataset._type', None, silent=True)
		if data is not None:
			terms.append(data)
		
		model = A.pull('info.model_type', '<>model._model_type', '<>model._type', None, silent=True)
		if model is not None:
			
			beta = A.pull('info.beta', '<>model.reg-wt', None)
			if beta is not None and beta > 0:
				# model = f'{model}{beta:2g}'.replace('.', 'p')
				model = f'{model}{beta}'.replace('.', 'p')
			terms.append(model)
		
		etype = A.pull('info.enc_type', None, silent=True)
		if etype is not None:
			terms.append(f'e-{etype}')
		dtype = A.pull('info.dec_type', None, silent=True)
		if dtype is not None:
			terms.append(f'd-{dtype}')
		if etype is None and dtype is None:
			atype = A.pull('info.arch_type', None, silent=True)
			if atype is not None:
				terms.append(atype)
		

		extra = A.pull('info.extra', None, silent=True)
		if extra is not None:
			terms.append(str(extra))

		name = '_'.join(terms)
		
		if not len(name):
			name = 'unknown'

		return name.replace(' ', '')
