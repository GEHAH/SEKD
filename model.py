import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from model_block.EGNN import *
import warnings

warnings.filterwarnings("ignore")


class KD_EGNN(nn.Module):
	def __init__(self, infeature_size, outfeature_size, nhidden_eg, edge_feature,
				 n_eglayer, nclass, device):
		super(KD_EGNN,self).__init__()
		self.dropout = 0.3
		self.eg1 = eg(in_node_nf=infeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=outfeature_size,
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True,
					  device=device)
		self.eg2 = eg(in_node_nf=outfeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=outfeature_size,
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True,
					  device=device)
		self.eg3 = eg(in_node_nf=outfeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=int(outfeature_size / 2),
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True,
					  device=device)
		self.eg4 = eg(in_node_nf=int(outfeature_size / 2),
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=int(outfeature_size / 4),
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True,
					  device=device)
		self.fc1 = nn.Sequential(
			nn.Linear(outfeature_size, int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)
		self.fc2 = nn.Sequential(
			nn.Linear(outfeature_size, int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)
		self.fc3 = nn.Sequential(
			nn.Linear(int(outfeature_size / 2), int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)
		self.fc4 = nn.Sequential(
			nn.Linear(int(outfeature_size / 4), int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)

	def forward(self,x_res,x_pos,edge_index):

		x_res = F.dropout(x_res, self.dropout, training=self.training)
		output_res, pre_pos_res = self.eg1(h=x_res,
										   x=x_pos.float(),
										   edges=edge_index,
										   edge_attr=None)

		output_res2, pre_pos_res2 = self.eg2(h=output_res,
											 x=pre_pos_res.float(),
											 edges=edge_index,
											 edge_attr=None)

		output_res3, pre_pos_res3 = self.eg3(h=output_res2,
											 x=pre_pos_res2.float(),
											 edges=edge_index,
											 edge_attr=None)

		output_res4, pre_pos_res4 = self.eg4(h=output_res3,
											 x=pre_pos_res3.float(),
											 edges=edge_index,
											 edge_attr=None)
		out1 = self.fc1(output_res)
		out2 = self.fc2(output_res2)
		out3 = self.fc3(output_res3)
		out4 = self.fc4(output_res4)

		return [out4,out3,out2,out1],[output_res4,output_res3,output_res2,output_res]


class EGNN(nn.Module):
	def __init__(self, infeature_size, outfeature_size, nhidden_eg, edge_feature,
				 n_eglayer, nclass, device):
		super(EGNN,self).__init__()
		self.dropout = 0.3
		self.eg1 = eg(in_node_nf=infeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=outfeature_size,
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True,
					  device=device)
		self.eg2 = eg(in_node_nf=outfeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=outfeature_size,
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True,
					  device=device)
		self.eg3 = eg(in_node_nf=outfeature_size,
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=int(outfeature_size / 2),
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True,
					  device=device)
		self.eg4 = eg(in_node_nf=int(outfeature_size / 2),
					  nhidden=nhidden_eg,
					  n_layers=n_eglayer,
					  out_node_nf=int(outfeature_size / 4),
					  in_edge_nf=edge_feature,
					  attention=True,
					  normalize=False,
					  tanh=True,
					  device=device)


		self.fc4 = nn.Sequential(
			nn.Linear(int(outfeature_size / 4), int(outfeature_size / 4)),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(int(outfeature_size / 4), nclass)
		)

	def forward(self,x_res,x_pos,edge_index):

		x_res = F.dropout(x_res, self.dropout, training=self.training)
		output_res, pre_pos_res = self.eg1(h=x_res,
										   x=x_pos.float(),
										   edges=edge_index,
										   edge_attr=None)

		output_res2, pre_pos_res2 = self.eg2(h=output_res,
											 x=pre_pos_res.float(),
											 edges=edge_index,
											 edge_attr=None)

		output_res3, pre_pos_res3 = self.eg3(h=output_res2,
											 x=pre_pos_res2.float(),
											 edges=edge_index,
											 edge_attr=None)

		output_res4, pre_pos_res4 = self.eg4(h=output_res3,
											 x=pre_pos_res3.float(),
											 edges=edge_index,
											 edge_attr=None)
		out4 = self.fc4(output_res4)

		return out4
