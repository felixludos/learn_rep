import numpy as np
import torch
from .generator import Generator
import networkx as nx 

class ER(Generator):
	def __init__(self, num_nodes, exp_edges = 1, noise_type='isotropic-gaussian', noise_sigma = 1.0, num_samples=1000, seed = 10):
		self.noise_sigma = noise_sigma
		if num_nodes == 2:
			exp_edges = 0.8
		p = float(exp_edges)/ (num_nodes-1)
		
		acyclic = 0
		count = 1

		while not acyclic:
			if exp_edges <= 2:
				self.graph = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes, p, directed = True, seed = seed*count)
			else:
				self.graph = nx.generators.random_graphs.gnp_random_graph(num_nodes, p, directed = True, seed = seed*count)
			acyclic = expm_np(nx.to_numpy_matrix(self.graph), num_nodes) == 0
			count += 1
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

	def sample(self, num_samples, graph = None, node = None, value = None):
		if graph is None:
			graph = self.graph

		samples = torch.zeros(num_samples, self.num_nodes)
		edge_pointer = 0
		for i in nx.topological_sort(graph):
			if i == node:
				noise = torch.tensor([value]*num_samples)
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

	def intervene(self, num_samples, node = None, value = None):
		if node is None:
			node = torch.randint(self.num_nodes, (1,))
		if value is None:
			value = torch.distributions.uniform.Uniform(-5,5).sample()

		mutated_graph = self.adjacency_matrix.copy()
		mutated_graph[:, node] = 0. #Cut off all the parents

		return self.sample(num_samples, nx.DiGraph(mutated_graph), node.item(), value.item()), node, value

def matrix_poly_np(matrix, d):
    x = np.eye(d) + matrix/d
    return np.linalg.matrix_power(x, d)

def expm_np(A, m):
    expm_A = matrix_poly_np(A, m)
    h_A = np.trace(expm_A) - m
    return h_A
