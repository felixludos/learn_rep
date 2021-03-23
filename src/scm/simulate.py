# import torch
from .data import erdos_renyi, dimlift
from .metrics import shd
class SCM_Simul:
	
	def __init__(self, num_nodes, num_samples, device = "cuda"):
		
		super().__init__()
		
		self.er = erdos_renyi.ER(num_nodes, num_samples=num_samples)
		ld = dimlift.LiftDimNonLinear(self.er.samples, device = device).to(device)
		ld.trained = 1 # skip training for now
		self.gt_graph = self.er.adjacency_matrix #Provides ground truth adjacency matrix
		self.samples = ld.collect_samples() #high dimensional samples
	
	def __getitem__(self, item):
		
		return self.samples[item]

if __name__ == '__main__':
	temp = SCM_Simul(5)
	print(temp.gt_graph)
	print(shd.shd(temp.gt_graph, temp.gt_graph)) #Should print SHD 0