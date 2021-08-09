import numpy as np
import torch
from .generator import Generator
import networkx as nx

from omnilearn import util

class ER(Generator):
	def __init__(self, num_nodes, exp_edges = 1, noise_type='isotropic-gaussian', noise_sigma = 1.0, num_samples=1000, seed = 10):
		self.noise_sigma = noise_sigma
		if num_nodes == 2:
			exp_edges = 0.8
		p = float(exp_edges)/ (num_nodes-1)
		
		acyclic = 0

		while not acyclic:
			seed = util.gen_deterministic_seed(seed)
			if exp_edges <= 2:
				self.graph = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes, p, directed = True, seed = seed)
			else:
				self.graph = nx.generators.random_graphs.gnp_random_graph(num_nodes, p, directed = True, seed = seed)
			acyclic = expm_np(nx.to_numpy_matrix(self.graph), num_nodes) == 0
		super().__init__(num_nodes, len(self.graph.edges), noise_type, num_samples, seed = seed)
		self.init_sampler()
		self.samples = self.sample(self.num_samples)

	def __getitem__(self, index):
		return self.samples[index]

	def build_graph(self):
		self.adjacency_matrix = nx.to_numpy_matrix(self.graph)
		self.weighted_adjacency_matrix = self.adjacency_matrix.copy()
		edge_pointer = 0
		for i in nx.topological_sort(self.graph):
			parents = list(self.graph.predecessors(i))
			if len(parents) == 0:
				continue
			else:
				for j in parents:
					self.weighted_adjacency_matrix[j, i] = self.weights[edge_pointer]
					edge_pointer += 1

	def init_sampler(self):
		if self.noise_type.endswith('gaussian'):
			#Identifiable
			if self.noise_type == 'isotropic-gaussian':
				noise_std= [self.noise_sigma]*self.num_nodes
			elif self.noise_type == 'gaussian':
				noise_std = np.linspace(0.1, 3., self.num_nodes)
			for i in range(self.num_nodes):
				self.graph.nodes[i]['sampler'] = torch.distributions.normal.Normal(0., noise_std[i])

		elif self.noise_type == 'exponential':
			noise_std= [self.noise_sigma]*self.num_nodes
			for i in range(self.num_nodes):
				self.graph.nodes[i]['sampler'] = torch.distributions.exponential.Exponential(noise_std[i])

	# def sample(self, num_samples, graph = None, node = None, value = None):
	# 	if graph is None:
	# 		graph = self.graph
	#
	# 	samples = torch.zeros(num_samples, self.num_nodes)
	# 	edge_pointer = 0
	# 	for i in nx.topological_sort(graph):
	# 		if i == node:
	# 			noise = torch.tensor([value]*num_samples)
	# 		else:
	# 			noise = self.graph.nodes[i]['sampler'].sample([num_samples])
	# 		parents = list(self.graph.predecessors(i))
	# 		if len(parents) == 0:
	# 			samples[:,i] = noise
	# 		else:
	# 			curr = 0.
	# 			for j in parents:
	# 				curr += self.weighted_adjacency_matrix[j, i]*samples[:,j]
	# 				edge_pointer += 1
	# 			curr += noise
	# 			samples[:, i] = curr
	# 	return samples
	

	def sample(self, num_samples, graph = None, values = {}):
		if graph is None:
			graph = self.graph

		samples = torch.zeros(num_samples, self.num_nodes)
		edge_pointer = 0
		for i in nx.topological_sort(graph):
			if i in values:
				noise = torch.tensor([values[i]]*num_samples)
			else:
				noise = self.graph.nodes[i]['sampler'].sample([num_samples])
			parents = list(self.graph.predecessors(i))
			if len(parents) == 0:
				samples[:,i] = noise
			else:
				curr = 0.
				for j in parents:
					curr += self.weighted_adjacency_matrix[j, i]*samples[:,j]
					edge_pointer += 1
				curr += noise
				samples[:, i] = curr
		return samples

	def _cut_graph(self, node, graph=None):
		if graph is None:
			graph = self.adjacency_matrix.copy()
		graph[:, node] = 0. #Cut off all the parents
		return graph

	# def intervene(self, num_samples, node = None, value = None):
	# 	if node is None:
	# 		node = torch.randint(self.num_nodes, (1,))
	# 	if value is None:
	# 		value = torch.distributions.uniform.Uniform(-5,5).sample()
	#
	# 	mutated_graph = self._cut_graph(node)
	#
	# 	return self.sample(num_samples, nx.DiGraph(mutated_graph), {node.item():value.item()}), node, value

	def intervene(self, num_samples, node = None, value = None):
		return self.multi_intervene(num_samples, node, value)

	def multi_intervene(self, num_samples, nodes, values=None):
		'''
		
		:param num_samples:
		:param nodes: list of indices
		:param values: list of values for each node that should be intervened
		:return:
		'''
		if not isinstance(nodes, (list, tuple)):
			nodes = [nodes]
		if values is None:
			values = torch.rand(len(nodes)).mul(10).sub(5)
		elif not isinstance(values, (list, tuple)):
			values = [values]
		assert len(values) == len(nodes)
		
		graph = None
		for node in nodes:
			graph = self._cut_graph(node, graph)
		
		return self.sample(num_samples, nx.DiGraph(graph), dict(zip(nodes, values))), values

		

