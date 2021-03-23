import torch
import torch.nn as nn
import tqdm

class LiftDimNonLinear(nn.Module):
	def __init__(self, samples, hidden_size=[100], out_scale = 100, batch_size = None, device = "cuda"):
		super().__init__()
		num_nodes = samples.shape[-1]
		if batch_size is None:
			batch_size = samples.shape[0]
		self.input_shape = num_nodes
		self.out_shape =  num_nodes * out_scale

		self.num_nodes = num_nodes
		self.device = device
		layers = []
		layers += [nn.Linear(self.input_shape, hidden_size[0]), nn.ReLU()]
		for i in range(len(hidden_size)-1):
			layers += [nn.Linear(hidden_size[i], hidden_size[i+1]), nn.ReLU()]
		layers += [nn.Linear(hidden_size[-1],self.out_shape)]
		self.network = nn.Sequential(*layers)
		self.mu_prior = torch.distributions.uniform.Uniform(-50, 50).sample([self.out_shape])
		self.loss_fn = nn.MSELoss(reduction = 'sum') 
		self.loader = torch.utils.data.DataLoader(samples, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
		self.trained = 0
	
	def forward(self, input):
		return self.network(input)
		

	def loss(self, pred):
		ns = pred.shape[0]
		target = self.mu_prior.expand_as(pred)
		mse = self.loss_fn(pred, target.to(self.device))
		return mse/ns

	def train(self, num_epochs = 1000):
		if self.trained:
			return 0
		optimizer = torch.optim.Adam(self.network.parameters() , 0.01)
		for e in tqdm.tqdm(range(num_epochs)):
			for data in self.loader:
				optimizer.zero_grad()
				out = self.forward(data.to(self.device))
				loss = self.loss(out)
				loss.backward()
				optimizer.step()
		self.trained = 1

	def collect_samples(self):
		if not self.trained:
			self.train()
		data_out = []
		curr = 0
		for data in self.loader:
			data_out.append(self.forward(data.to(self.device)).cpu())
		data_out = torch.cat(data_out, dim = 0)
		return data_out

	
