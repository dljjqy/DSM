import sys
sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from BaseTrainer import BaseTrainer
from MultiCofDs import ConvC1Ds, ConvC3Ds, ConvRandC3Ds
from MyPlot import multi_cof_draw_img as draw_img
from Generators import ConvJac

class ConvTrainer(BaseTrainer):
	def __init__(self, max_iter=5, *args, **kwargs):
		self.max_iter = max_iter
		super().__init__(*args, **kwargs)
		self.h = 1.0 / self.GridSize
		self.generator = ConvJac(self.dtype, self.device, self.GridSize, self.h)
	
	@property
	def name(self):
		return f"{self.tag}"

	def reboot(self):
		self.init_traindl()
		self.init_valdl()
		self.config_optimizer(self.lr)
	
	def hyper_param_need2save(self):
		kwargs = {
			'GridSize': self.GridSize,
			'area': self.area,
			'trainN': self.trainN,
			'valN': self.valN,
			'Batch_size': self.batch_size,
			'lr':self.lr,
			'epochs':self.total_epochs,
			'method': 'ConvJac',
			'max_iter': self.max_iter,
			'name': self.name,
			'tag':self.tag,
			'net_kwargs': self.net_kwargs
		}
		return kwargs

	def init_traindl(self):
		self.start=1
		match self.net_kwargs['in_channels']:
			case 1:
				train_ds = ConvC1Ds(self.start, self.trainN, self.area, self.GridSize, self.dtype, self.device)
			case 3:
				train_ds = ConvC3Ds(self.start, self.trainN, self.area, self.GridSize, self.dtype, self.device)
		
		self.train_dl = DataLoader(train_ds, self.batch_size, shuffle=True, num_workers=0)
	
	def init_valdl(self):
		self.start=1
		match self.net_kwargs['in_channels']:
			case 1:
				val_ds = ConvC1Ds(self.start + self.trainN, self.valN, self.area, self.GridSize, self.dtype, self.device)
			case 3:
				val_ds = ConvC3Ds(self.start + self.trainN, self.valN, self.area, self.GridSize, self.dtype, self.device)
		self.val_dl = DataLoader(val_ds, self.batch_size)
	
	def predict(self, data):
		# print(data.shape)
		return self.net(data)
	
	def train_step(self, data, K, b, max_iter):
		pre = self.predict(data)

		with torch.no_grad():
			labels = self.generator(max_iter, pre, b, K)

		loss = self.loss_fn(labels, pre)
			
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		loss_val = loss.item()
		self.writer.add_scalar("Train-SubIterLoss", loss_val, self.train_global_idx)
		
		self.train_global_idx += 1
		return  loss_val

	def train_loop(self):
		self.net.train()
		loss_vals = []
		for data, K, _, b in tqdm(self.train_dl, desc='Training Loop:', position=1, leave=False):
			loss_val = self.train_step(data, K, b, np.random.randint(1,self.max_iter))
			loss_vals.append(loss_val)
		self.lr_scheduler.step()

		return np.array(loss_vals).mean()
	
	def val_step(self, data, u):
		pre = self.predict(data)
		val_real_loss = self.loss_fn(u, pre).item()
		self.writer.add_scalar("Val-RealLoss", val_real_loss, self.val_global_idx)
		
		self.val_global_idx += 1
		return pre, val_real_loss

	def val_plot(self, pre, U, cofs):
		batch_size = U.shape[0]

		pre = pre.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
		sols = U.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
		cofs = cofs.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)

		k = np.random.choice(batch_size)
		fig = draw_img('Validation', pre[k], sols[k], cofs[k], self.GridSize, a=1)
		self.writer.add_figure(f"ValFigure", fig, self.val_global_idx)
	
	def val_loop(self):
		self.net.eval()
		real_loss_vals = []
		for data, K, u, b in tqdm(self.val_dl, desc='Validation Loop:', position=2, leave=False):
			pre, real_loss = self.val_step(data, u)

			real_loss_vals.append(real_loss)
			self.writer.add_scalar("Val-RealLoss", real_loss, self.val_global_idx)
			
		self.val_plot(pre, u, K)
		return np.array(real_loss_vals).mean(), np.array(real_loss_vals).mean()

if __name__ == '__main__':
	GridSize = 96
	tag = '96UNetC3Tanh-ConvJac5-batch12-Double'
	trainer = ConvTrainer(
		max_iter=5,
		area = ((0, 0), (1, 1)),
		GridSize=GridSize,
		trainN=10000,
		valN=100,
		batch_size=12,
		net_kwargs={
			'model_name': 'UNet',
			'Block': "ResBottleNeck",
			'planes':8,
			'in_channels':3,
			'classes':1,
			'GridSize':GridSize,
			'layer_nums':   [2,2,2,2],
			'factor':2,
			'norm_method': 'batch',
			'pool_method':'max',
			'padding':'same',
			'padding_mode':'reflect',
			'end_padding':'same',
			'end_padding_mode':'reflect',
			'act': 'tanh',
		},
		log_dir=f'./all_logs',
		lr=1e-3,
		loss_fn=torch.nn.functional.mse_loss,
		model_save_path=f'./model_save',
		tag = tag,
		total_epochs=[150],
		device='cuda',
		dtype=torch.double,
		hyper_params_save_path=f'./hyper_parameters'
	)
	trainer.fit_loop()
	
