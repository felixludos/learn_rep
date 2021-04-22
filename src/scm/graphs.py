
import omnifig as fig
import random
import networkx as nx
import torch
from torch import nn
from torch import distributions as distrib
import numpy as np
import zlib
from omnibelt import unspecified_argument

from omnilearn import util
from omnilearn import Function

# PRESETS = ['chain', 'collider','fork', 'random']
# NOISE_TYPES = ['gaussian', 'isotropic-gaussian', 'exponential', 'gumbel']
# VARIABLE_TYPES = ['gaussian', 'non-gaussian', 'categorical']

@fig.Component('scm')
class StructuralModel(util.Seed, Function):
	
	def __init__(self, A, samplers=unspecified_argument, num_nodes=unspecified_argument,
	             exp_edges=unspecified_argument,
	             din=None, dout=None,
	             default_sampler=None, **kwargs):
		
		if exp_edges is unspecified_argument:
			exp_edges = A.pull('exp-edges', 1.)
		
		if samplers is unspecified_argument:
			samplers = A.pull('samplers', None)
		if samplers is None:
			if num_nodes is unspecified_argument:
				num_nodes = A.pull('num-nodes')
			
			if default_sampler is None:
				default_sampler = A.pull('default-sampler', 'normal')
			if default_sampler == 'normal':
				samplers = [distrib.Normal(0.,1.) for _ in range(num_nodes)]
			elif default_sampler == 'uniform':
				samplers = [distrib.Uniform(-1., 1.) for _ in range(num_nodes)]
			else:
				raise NotImplementedError
			
		num_nodes = len(samplers)
		assert num_nodes > 1, f'Bad number of nodes: {num_nodes}'
		
		# TODO: config to specify graph
		
		if din is None:
			din = num_nodes
		if dout is None:
			dout = num_nodes
		
		super().__init__(A, din=din, dout=dout, **kwargs)
		
		if num_nodes == 2:
			exp_edges = 0.8
		p = exp_edges / (num_nodes - 1)
		
		acyclic = False
		seed = self.seed
		while not acyclic:
			seed = util.gen_deterministic_seed(seed)
			graph = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes, p, directed=True, seed=seed) \
				if exp_edges <= 2 else nx.generators.random_graphs.gnp_random_graph(num_nodes,
				                                                                    p, directed=True, seed=seed)
			acyclic = expm_np(nx.to_numpy_matrix(graph), num_nodes) == 0
		
		self.graph = graph
		
		self._node_order = list(nx.topological_sort(self.graph))
		self._node_ancestors = [list(self.graph.predecessors(node)) for node in self._node_order]
		
		self.num_nodes = num_nodes
		self.samplers = samplers
		
		self.equations = nn.ModuleList(self._build_equations(A, graph))
	
	def _initialize_equation(self, equation):
		
		for param in equation.parameters():
			param.requires_grad = False
			param.data.normal_()
			
		return equation
	
	def _build_equations(self, A, graph):
		
		use_init = A.pull('use-initializer', True)
		
		A.push('equation._type', 'mlp', silent=True)
		eq = A.pull('equation', raw=True)
		eq.push('dout', 1, silent=True)
		
		eqs = [None]*len(self._node_order)
		
		for node in self._node_order:
			din = 1 + len(self._node_ancestors[node])
			eq.push('din', din, silent=True)
			equation = eq.pull_self()
			if use_init:
				equation = self._initialize_equation(equation)
				
			eqs[node] = equation
		
		return eqs
		

	def sample(self, batch_size=None, hard={}, soft={}, ret_noise=False):
		if batch_size is None:
			if len(hard):
				batch_size = next(iter(hard.values())).size(0)
			if len(soft):
				batch_size = next(iter(soft.values())).size(0)
		B = 1 if batch_size is None else batch_size

		x = torch.zeros(B, self.num_nodes)
		n = {} if ret_noise else None
		
		for node in self._node_order:
			
			if node in hard:
				x[:,node] = hard[node]
			else:
				noise = soft[node] if node in soft else self.samplers[node].sample((B,))
				noise = noise.reshape(B, -1)
				if n is not None:
					n[node] = noise.view(B)
				if len(self._node_ancestors[node]):
					noise = torch.cat([noise, x[:,self._node_ancestors[node]]], -1)
				x[:,node] = self.equations[node](noise).squeeze(-1)
				
		if batch_size is None:
			x = x.squeeze(0)
		if ret_noise:
			return x, n
		return x

def matrix_poly_np(matrix, d):
	x = np.eye(d) + matrix/d
	return np.linalg.matrix_power(x, d)

def expm_np(A, m):
	expm_A = matrix_poly_np(A, m)
	h_A = np.trace(expm_A) - m
	return h_A
