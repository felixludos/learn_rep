import omnifig as fig
import foundation as fd

@fig.Component('sae-run')
class SAE_Run(fd.op.TorchRun):

	def _gen_name(self, A):

		model = A.pull('info.model_type', '<>model._model_type', '<>model._type', None, silent=True)
		data = A.pull('info.dataset_type', '<>dataset.name', '<>dataset._type', None, silent=True)

		name = f'{model}_{data}'

		arch = A.pull('info.arch', None, silent=True)
		if arch is not None:
			name = f'{name}_{arch}'

		extra = A.pull('info.extra', None, silent=True)
		if extra is not None:
			name = f'{name}_{extra}'

		return name
