
# # @fig.AutoModifier('teval')
# class Transfer_Eval(fd.Model):
#
# 	def __init__(self):
# 		raise NotImplementedError
#
# 	def prep(self, *datasets):
# 		dataset = datasets[0]
# 		assert isinstance(dataset, transfer.Multi_Dataset), f'not transfer setting: {dataset}'
# 		super().prep(*datasets)
#
# 	def _evaluate(self, info):
#
# 		A = info['A']
#
# 		valdata = info['datasets'][-1]
# 		testdata = None
# 		if 'testset' in info:
# 			testdata = info['testset']
#
# 		groups = {name: [name] for name in valdata.folds}
# 		full = '_full' if 'full' in groups else 'full'
# 		groups[full] = list(groups)
# 		groups = A.pull('groups', groups)
#
# 		print('Will run the full evaluation {} times (once per group)'.format(len(groups)))
#
# 		logger = info['logger']
# 		logger_id = info['identifier']
#
# 		results = {}
#
# 		for name, folds in groups.items():
#
# 			print('Evaluation for group {}: {}'.format(name, ', '.join(folds)))
#
# 			if logger is not None:
# 				logger.set_tag_format('{}-{}/{}'.format(logger_id, name, '{}'))
#
# 			valdata.set_full(*folds)
# 			if testdata is not None:
# 				testdata.set_full(*folds)
#
# 			results[name] = super()._evaluate(info)
#
# 		return results
#

# class Cost_Aware(Prior_Autoencoder):
# 	def __init__(self, A):
#
# 		reg_imp_p = A.pull('reg_imp_p', 1)
#
# 		reg_imp_wt = A.pull('reg_imp_wt', 0.5)
# 		reg_prior_wt = A.pull('reg_prior_wt', 1)
#
# 		init_imp_mu = A.pull('init_imp_mu', 0)
# 		init_imp_std = A.pull('init_imp_std', 1)
#
# 		weigh_distances = A.pull('weigh_distances', False)
#
# 		imp_noise = A.pull('imp_noise', 0)
#
# 		super().__init__(A)
#
# 		self.register_cache('_rew_q')
#
# 		self.stats.new('imp', 'reg_imp', 'reg_prior')
#
# 		self.importance = nn.Parameter(init_imp_std*torch.randn(self.latent_dim) + init_imp_mu,
# 		                               requires_grad=True)
#
# 		self.reg_imp_p = reg_imp_p
#
# 		self.reg_imp_wt = reg_imp_wt
# 		self.reg_prior_wt = reg_prior_wt
#
# 		self.weigh_distances = weigh_distances
# 		self.imp_noise = imp_noise
#
# 	def get_importance(self, noisy=False):
# 		imp = self.importance
# 		if noisy and self.imp_noise > 0:
# 			imp = imp + torch.randn_like(imp).mul(self.imp_noise)
# 		return F.sigmoid(imp)
#
# 	def _visualize(self, info, logger):
#
# 		if self._viz_counter % 2 == 0 or not self.training:
# 			logger.add('histogram', 'imp_hist', self.importance.clamp(min=-5, max=5))
# 			logger.add('text', 'imp_str', '[{}]'.format(', '.join('{:2.3f}'.format(i.item()) for i in self.importance)))
#
# 		super()._visualize(info, logger)
#
# 	def regularize(self, q):
#
# 		v = self.get_importance()
# 		self.stats.update('imp', v.sum())
#
# 		# reg_imp = v.norm(p=self.reg_imp_p)
# 		# reg_imp = F.elu(self.importance).sum()
# 		reg_imp = v.pow(self.reg_imp_p).sum()
# 		self.stats.update('reg_imp', reg_imp)
#
# 		if self._raw_q is not None:
# 			q = self._raw_q
# 			self._raw_q = None
#
# 		p = self.sample_prior(q.size(0))
# 		if self.weigh_distances:
# 			q = q * v.unsqueeze(0).detach()
# 			p = p * v.unsqueeze(0).detach()
#
# 		reg_prior = super().regularize(q, p)
# 		self.stats.update('reg_prior', reg_prior)
#
# 		return self.reg_imp_wt * reg_imp + self.reg_prior_wt * reg_prior


# @fig.Component('cae')
# class Det_Cost_Aware(Cost_Aware):
# 	def encode(self, x):
# 		q = super().encode(x)
# 		B, D = q.size()
#
# 		self._raw_q = q
#
# 		v = self.get_importance(noisy=True).expand(B, D)
# 		p = self.sample_prior(B)
#
# 		q = v * q + (1 - v) * p
# 		# q = q + (1 - v) * p
# 		return q
#
# class Sto_Cost_Aware(Cost_Aware):
# 	def encode(self, x):
# 		q = super().encode(x)
# 		return self.as_normal(q)
#
# 	def as_normal(self, q):
# 		std = self.get_importance(noisy=True).expand(*q.size())
# 		return distrib.Normal(loc=q, scale=std)
#
# @fig.Component('cwae')
# class Cost_Aware_WAE(Det_Cost_Aware, WAE):
# 	pass
#
# @fig.Component('cswae')
# class Cost_Aware_SWAE(Det_Cost_Aware, Slice_WAE):
# 	pass
#
# @fig.Component('cvae')
# class Cost_VAE(Sto_Cost_Aware, VAE):
# 	pass


# @fig.AutoModifier('fixed-std')
# class Fixed_Std(fd.Visualizable, fd.FunctionBase):
# 	def __init__(self, A, latent_dim=None):
#
# 		if latent_dim is None:
# 			latent_dim = A.pull('latent_dim', '<>dout')
#
# 		min_log_std = A.pull('min_log_std', None)
#
# 		super().__init__(A)
#
# 		self.log_std = nn.Parameter(torch.randn(latent_dim)*0.1, requires_grad=True)
#
# 		self.min_log_std = min_log_std
# 		self.latent_dim = latent_dim
#
# 	def get_hparams(self):
# 		return {'std_type': 'fixed'}
#
# 	def _visualize(self, info, logger):
#
# 		try:
# 			super()._visualize(info, logger)
# 		except NotImplementedError:
# 			pass
#
#
# 		pass
#
# 	def forward(self, *args, **kwargs):
#
# 		mu = super().forward(*args, **kwargs)
# 		logsigma = self.log_std
#
# 		if self.min_log_std is not None:
# 			logsigma = logsigma.clamp(min=self.min_log_std)
#
# 		return distrib.Normal(loc=mu, scale=logsigma.exp())





# endregion


# region Architectures


# @fig.Component('extraction-enc')
# class UMAP_Encoder(fd.Encodable, fd.FunctionBase):
#
# 	def __init__(self, A):
#
# 		in_shape = A.pull('in_shape', '<>din')
# 		latent_dim = A.pull('latent_dim', '<>dout')
# 		feature_dim = A.pull('feature_dim', '<>latent_dim')
#
# 		transform = A.pull('transform', None)
#
# 		alg = A.pull('alg', 'umap')
#
# 		kwargs = {
# 			'n_components': feature_dim,
# 		}
#
# 		if alg == 'umap':
#
# 			extraction_cls = gpumap.GPUMAP
#
# 			kwargs['random_state'] = A.pull('random_state', '<>seed')
# 			kwargs['min_dist'] = A.pull('min_dist', 0.1)
# 			kwargs['n_neighbors'] = A.pull('neighbors', 15)
#
# 		elif alg == 'pca':
# 			extraction_cls = PCA
#
# 		else:
# 			raise Exception(f'unknown alg: {alg}')
#
# 		extractor = extraction_cls(**kwargs)
#
# 		if 'net' in A:
# 			A.net.din = feature_dim
# 			A.net.dout = latent_dim
#
# 		net = A.pull('net', None)
#
# 		training_limit = A.pull('training_limit', None)
#
# 		super().__init__(din=in_shape, dout=feature_dim if net is None else latent_dim)
#
# 		self.training_limit = training_limit
#
# 		self.transformer = transform
#
# 		self.alg = alg
# 		self.extractor = extractor
#
# 		self.net = net
#
# 		# self.set_optim(A)
# 		# self.set_scheduler(A)
#
# 	def _resize(self, x):
# 		N, C, H, W = x.shapes
#
# 		if H >= 64:
# 			return x[:, :, ::2, ::2].reshape(N, -1)
# 		return x.reshape(N, -1)
#
# 	def prep(self, traindata, *other):
#
# 		samples = traindata.get_raw_data().float()
#
# 		if self.training_limit is not None:
# 			samples = samples[:self.training_limit]
#
# 		samples = self._reformat(samples)
#
# 		print(f'Training a {self.alg} feature extractor to extract {self.extractor.n_components} '
# 		      f'features from an input {samples.shape}')
#
#
# 		# fit estimator
# 		self.extractor.fit(samples)
#
# 		print('Feature extraction complete')
#
# 	def encode(self, x):
# 		return self(x)
#
# 	def transform(self, x):
#
# 		device = x.device
# 		x = self._reformat(x)
#
# 		q = self.extractor.transform(x)
# 		q = torch.from_numpy(q).to(device)
#
# 		return q
#
# 	def _reformat(self, x):
# 		x = x.cpu().numpy()
#
# 		if self.transformer is not None:
# 			x = self.transformer(x)
# 		else:
# 			x = self._resize(x)
#
# 		return x
#
# 	def forward(self, x):
#
# 		q = self.transform(x)
#
# 		if self.net is None:
# 			return q
# 		return self.net(q)



# def get_name(A):
# 	if 'name' not in A:
# 		model, data = None, None
# 		arch = None
# 		if 'info' in A:
# 			if 'model_type' in A.info:
# 				model = A.info.model_type
# 			if 'dataset_type' in A.info:
# 				data = A.info.dataset_type
# 			if 'arch' in A.info:
# 				arch = A.info.arch
# 		if model is None:
# 			model = A.model._type
# 		if data is None:
# 			if 'name' in A.dataset:
# 				data = A.dataset.name
# 			else:
# 				data = A.dataset._type.split('-')[-1]
# 		name = '{}_{}'.format(model,data)
# 		if arch is not None:
# 			name = '{}_{}'.format(name, arch)
#
# 	if 'info' in A and 'extra' in A.info:
# 		name = '{}_{}'.format(name, A.info.extra)
#
# 	return name


