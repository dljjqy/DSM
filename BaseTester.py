import torch
import json
import numpy as np
import pandas as pd

from SegModel import CNN

class BaseTester:
	def __init__(
				self, 
				GridSize,
				area,
				ckpt_save_path,
				hyper_parameters_save_path, 
				img_save_path, 
				device='cuda',
				dtype=torch.float):
		self.GridSize = GridSize
		self.ckpt_save_path = f"{ckpt_save_path}/{GridSize}"
		self.hyper_parameters_save_path = f"{hyper_parameters_save_path}/{GridSize}"
		self.device = device
		self.dtype = dtype
		self.area = area
		self.h = (area[1][0] - area[0][0]) /self.GridSize

		self.img_save_path = f"{img_save_path}/{GridSize}"
	
	def l2(self, pre, ans):
		return np.sqrt(((pre - ans)**2 * self.h**2).sum())
			
	def load_kwargs(self, exp_name):
		with open(f"{self.hyper_parameters_save_path}/{exp_name}.json") as file:
				self.kwargs = json.load(file)

	def load_ckpt(self, best_or_last, exp_name):
		self.load_kwargs(exp_name)
		ckpt = torch.load(f"{self.ckpt_save_path}/{exp_name}/{best_or_last}.pt")

		net_kwargs = self.kwargs['net_kwargs'].copy()
		model_name = net_kwargs.pop('model_name')
		self.net = CNN(model_name, net_kwargs, self.dtype, self.device)
	
		self.net.load_state_dict(ckpt)
		self.net.eval()

	def mesh(self):
		(self.left, self.bottom), (self.right, self.top) = self.area
		self.dx = (self.right - self.left) / self.GridSize
		self.dy = (self.top - self.bottom) / self.GridSize

		self.xx, self.yy = np.meshgrid(
				np.arange(self.left + self.dx/2, self.right, self.dx),
				np.arange(self.bottom + self.dy/2, self.top, self.dy)) 
	
	def l2_csv(self, pre, ans):
		df = {
			'id':[],
			'l2':[]
		}
		DataN = len(pre)
		for i in range(DataN):
			l2_error = self.l2(pre[i], ans[i])
			df['id'].append(i)
			df['l2'].append(l2_error)
		return pd.DataFrame(df)