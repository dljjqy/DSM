# Used to train CNN for solving the Poisson equation with parametric sources terms

import torch
import torch.nn as nn
from torch.nn.functional import pad as pad

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append('../')

from BaseTrainer import BaseTrainer
from Generators import ConvJac
from HeatMultibcDs import *

class ConvJac1(nn.Module):
	def __init__(self, dtype, device, GridSize):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.GridSize = GridSize

		self.gd = 298.0
		self.gn = 0.0
		self.K = torch.ones(GridSize, GridSize).to(dtype).to(device)
		with torch.no_grad():
			self.left, self.right, self.up, self.low, self.diag = self.compute_A()

	def forward(self, u, b, max_iter):
		with torch.no_grad():
			y = self.jac_step(u, b, )
			for _ in range(1, max_iter):
				y = self.jac_step(y, b)
			return y

	def compute_A(self):
		'''All edges are dirichlet
		'''
		y_lbd = pad(self.K, (0,0,1,1), 'constant', 0.0)
		up = 2 /  (y_lbd[..., 1:-1, :] + y_lbd[..., 2:, :])
		low = 2 / (y_lbd[..., 0:-2, :] + y_lbd[..., 1:-1, :])
		
		x_lbd = pad(self.K, (1,1,0,0), 'constant', 0.0)
		right =  2 / (x_lbd[..., :, 1:-1] + x_lbd[..., :, 2:])
		left = 2 / (x_lbd[..., :, 0:-2] + x_lbd[...,:, 1:-1])

		# print(up.shape, low.shape, left.shape, right.shape)

		diag = left + right + up + low
		return left, right, up, low, diag

	def jac_step(self, u, b):
		with torch.no_grad():
			y_pad_u = pad(u, (0,0,1,1), 'constant', self.gd)
			up_flux = -self.up * y_pad_u[..., 2:, :]
			low_flux = -self.low * y_pad_u[..., :-2, :]

			x_pad_u = pad(u, (1,1,0,0), 'constant', self.gd)
			left_flux = -self.left * x_pad_u[..., :, :-2]
			right_flux = -self.right * x_pad_u[..., :, 2:]

			# Jacobi
			pseudo_label = (b - (left_flux + right_flux + up_flux + low_flux)) / self.diag 
			return pseudo_label 

class ConvJac2(ConvJac1):
	def compute_A(self):
		'''Only left is Dirichlet
		'''		
		left_lbd = pad(self.K, (1, 0, 0, 0), 'constant', 0.0)
		left = 2 / (left_lbd[..., :, 0:-1] + left_lbd[...,:, 1:])
		
		right = 2 / (self.K[..., :, 0:-1,] + self.K[..., :, 1:])
		right = pad(right, (0, 1, 0, 0), 'constant')

		up = 2 / (self.K[..., 0:-1, :] + self.K[..., 1:, :])
		up = pad(up, (0, 0, 0, 1), 'constant')
		
		low = 2 / (self.K[...,0:-1,:] + self.K[...,1:,:])
		low = pad(low, (0, 0, 1, 0), 'constant')

		# print(up.shape, low.shape, left.shape, right.shape)
		diag = left + right + up + low
		return left, right, up, low, diag

	def jac_step(self, u, b):
		with torch.no_grad():
			low_flux = -self.low * pad(u, (0,0,1,0), 'constant', 0.0)[..., :-1, :]
			up_flux = -self.up * pad(u,   (0,0,0,1), 'constant', 0.0)[..., 1:, :]

			right_flux = -self.right * pad(u, (0,1,0,0), 'constant', 0.0)[..., :, 1:]
			
			left_flux = -self.left * (pad(u, (1,0,0,0),'constant', self.gd)[..., :, :-1])

			# Jacobi
			pseudo_label = (b - (left_flux + right_flux + up_flux + low_flux)) / self.diag 
			return pseudo_label 

class ConvJac3(ConvJac1):
	def compute_A(self):
		'''All are Neumann but a small window on the low side
		'''		
		h = 0.1/self.GridSize
		is_dirichlet = np.arange(h/2, 0.1, h)
		is_dirichlet = ((is_dirichlet <= 0.0505) * (is_dirichlet >= 0.0495)).astype(bool)

		left = 2 / (self.K[..., :, 0:-1] + self.K[..., :, 1:])
		left = pad(left, (1, 0, 0, 0), 'constant')
		
		right = 2 / (self.K[..., :, 0:-1,] + self.K[..., :, 1:])
		right = pad(right, (0, 1, 0, 0), 'constant')

		up = 2 / (self.K[..., 0:-1, :] + self.K[..., 1:, :])
		up = pad(up, (0, 0, 0, 1), 'constant')

		low = 2 / (self.K[..., 0:-1, :] + self.K[..., 1:, :])
		low = pad(low, (0, 0, 1, 0), 'constant')

		low[0][is_dirichlet] = 2.0
		diag = left + right + up + low
		return left, right, up, low, diag

	def jac_step(self, u, b):
		with torch.no_grad():
			left_flux = -self.left * pad(u,   (1,0,0,0), 'constant', 0.0)[..., :, :-1]
			right_flux = -self.right * pad(u, (0,1,0,0), 'constant', 0.0)[..., :, 1:]
			up_flux = -self.up * pad(u, (0,0,0,1), 'constant', 0.0)[..., 1:, :]

			low_flux = -self.low * pad(u, (0,0,1,0), 'constant', self.gd)[..., :-1, :]
			# Jacobi
			pseudo_label = (b - (left_flux + right_flux + up_flux + low_flux)) / self.diag 
			return pseudo_label 

class Trainer(BaseTrainer):
	def __init__(
		self, 
		max_iter, 
		*args, **kwargs
	):
		self.max_iter = max_iter
		super().__init__(*args, **kwargs)
		self.gen_mesh(self.area, self.GridSize)

		self.generators = [
			ConvJac1(self.dtype, self.device, self.GridSize),
			ConvJac2(self.dtype, self.device, self.GridSize),
			ConvJac3(self.dtype, self.device, self.GridSize)
		]
		
	@property
	def name(self):
		return f"{self.tag}"

	def reboot(self):
		self.config_optimizer(self.lr)

		self.init_traindl()
		self.init_valdl()

	def epoch_reboot(self, epoch_id):
		if (epoch_id+1) % 10 == 0:
			self.init_traindl()
			self.init_valdl()
		pass

	def gen_mesh(self, area, GridSize):
		(self.left, self.bottom), (self.right, self.top) = area
		self.bd_x, self.bd_y = np.meshgrid(
			np.linspace(self.left, self.right, GridSize),
			np.linspace(self.bottom, self.top, GridSize),
		)
		self.dx, self.dy = (self.right - self.left) / GridSize, (self.top - self.bottom) / GridSize
		self.xx, self.yy = np.meshgrid(
			np.arange(self.left + self.dx / 2, self.right, self.dx),
			np.arange(self.bottom + self.dy / 2, self.top, self.dy),
		)
		self.dx = (self.right - self.left) / GridSize
		self.dy = (self.top - self.bottom) / GridSize
	
	def hyper_param_need2save(self):
		param = {
			'GridSize': self.GridSize,
			# 'method': self.method,
			'max_iter': self.max_iter,
			'area': self.area,
			'trainN': self.trainN,
			'valN': self.valN,
			'Batch_size': self.batch_size,
			'lr':self.lr,
			'epochs':self.total_epochs,
			'tag':self.tag,
			'net_kwargs': self.net_kwargs,
		}
		return param

	def init_traindl(self):
		match self.net_kwargs['in_channels']:
			case 2:
				train_ds = C2TrainDS(self.GridSize, self.trainN, self.dtype, self.device, self.batch_size) 
			case 4:
				train_ds = C4TrainDs(self.GridSize, self.trainN, self.dtype, self.device, self.batch_size) 		
				# self.boundaries = c4_load_boundary(self.GridSize)
		self.train_dl = DataLoader(
			train_ds, self.batch_size, shuffle=False, drop_last=False,
		)
		# self.train_cases = train_ds.cases

	def init_valdl(self):
		match self.net_kwargs['in_channels']:
			case 2:
				val_ds = C2ValDS(self.GridSize, self.valN, self.dtype, self.device, self.batch_size) 
			case 4:
				val_ds = C4ValDs(self.GridSize, self.valN, self.dtype, self.device, self.batch_size) 

		self.val_dl = DataLoader(
			val_ds, self.batch_size, shuffle=False, drop_last=False
		)
		# self.val_cases = val_ds.cases

	def train_step(self, data, layouts, case_ids, max_iter):
		pre = self.net(data)
		
		# Generate the label by Jac
		with torch.no_grad():
			labels = []
			for i in range(self.batch_size):
				labels.append(self.generators[case_ids[i]-1](pre[i], layouts[i]*self.dx**2, max_iter))
			label = torch.stack(labels).type_as(pre)

		loss = self.loss_fn(pre, label)
		
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		loss_val = loss.item()
		self.writer.add_scalar("Train-PinnLoss", loss_val, self.train_global_idx)
		self.train_global_idx += 1
		return loss_val

	def train_loop(self):
		self.net.train()
		loss_vals = []
		for data, layout, case_id in tqdm(self.train_dl, desc='Training Loop:', leave=False, position=1):
			loss_val = self.train_step(data, layout, case_id, np.random.randint(1, self.max_iter))
			loss_vals.append(loss_val)
		self.lr_scheduler.step()
		return np.array(loss_vals).mean()

	def val_step(self, data, u):
		pre = self.net(data) 
		val_real_loss = self.loss_fn(pre, u).item()

		self.writer.add_scalar("Val-RealLoss", val_real_loss, self.val_global_idx)

		self.val_global_idx += 1
		return val_real_loss, pre

	def val_loop(self):
		self.net.eval()
		real_loss = []       
		for data, u in tqdm(self.val_dl, desc='Validation Loop', leave=False, position=2):
			val_real_loss, pre = self.val_step(data, u)
			real_loss.append(val_real_loss)
	
		self.val_plot(pre, u)
		return np.array(real_loss).mean(), np.array(real_loss).mean()

	def val_plot(self, pre, ans):
		pre = pre[0].cpu().numpy().reshape(self.GridSize, self.GridSize)
		ans = ans[0].cpu().numpy().reshape(self.GridSize, self.GridSize)
		fig = plt.figure()
		fig.suptitle('ValFigure', fontsize=20)
		fig.set_figheight(20)
		fig.set_figwidth(20)

		ax1 = fig.add_subplot(2, 2, 1, aspect="equal")
		ax2 = fig.add_subplot(2, 2, 2, aspect="equal")
		ax3 = fig.add_subplot(2, 2, 3, aspect="equal")
		ax4 = fig.add_subplot(2, 2, 4, aspect="equal")

		ax1.set_title(f'Prediction', fontsize=20)
		ctf = ax1.contourf(self.xx, self.yy, pre, alpha=1, cmap=cm.Spectral_r, levels=50)
		divider = make_axes_locatable(ax1)
		cax = divider.new_horizontal(size="2.5%", pad=0.1)
		fig.add_axes(cax)
		cbar = fig.colorbar(ctf, cax=cax, orientation="vertical")
		cbar.ax.tick_params(labelsize=20)

		ax2.set_title(f'Reference', fontsize=20)
		ctf = ax2.contourf(self.xx, self.yy, ans, alpha=1, cmap=cm.Spectral_r, levels=50)
		divider = make_axes_locatable(ax2)
		cax = divider.new_horizontal(size="2.5%", pad=0.1)
		fig.add_axes(cax)
		cbar = fig.colorbar(ctf, cax=cax, orientation="vertical")
		cbar.ax.tick_params(labelsize=20)

		ax3.set_title(f'Difference', fontsize=20)
		ctf = ax3.contourf(self.xx, self.yy, np.abs(ans - pre), alpha=1, cmap=cm.Spectral_r, levels=50)
		divider = make_axes_locatable(ax3)
		cax = divider.new_horizontal(size="2.5%", pad=0.1)
		fig.add_axes(cax)
		cbar = fig.colorbar(ctf, cax=cax, orientation="vertical")
		cbar.ax.tick_params(labelsize=20)

		levels = np.linspace(ans.min(), ans.max(), 10)[2:-2] 
		# levels=None
		ct1 = ax4.contour(self.xx, self.yy, pre, colors='r', linestyles='dashed', linewidths=1.5, levels=levels)
		ct2 = ax4.contour(self.xx, self.yy, ans, colors='b', linestyles='solid', linewidths=2, levels=levels)
		ax4.clabel(ct1, inline=False, fontsize=20)
		ax4.clabel(ct2, inline=False, fontsize=20)
		blue_line = mlines.Line2D([], [], color='blue', markersize=20, label='ref')
		red_line = mlines.Line2D([], [], color='red', markersize=20, label='pre')
		ax4.legend(handles=[blue_line, red_line], fontsize=20 )

		fig.tight_layout()
		self.writer.add_figure(f"ValFigure", fig, self.val_global_idx)
		plt.close(fig)


if __name__ == "__main__":
	from torch.nn.functional import mse_loss

	GridSize = 128
	tag = "C2_float"

	trainer = Trainer(
		max_iter=5,
		dtype=torch.double,
		device="cuda",
		area=((0, 0), (0.1, 0.1)),
		GridSize=GridSize,
		trainN=14000,
		valN=21,
		batch_size=7,
		net_kwargs={
			'model_name': 'UNet',
			'Block': "ResBottleNeck",
			'planes':6,
			'in_channels':2,
			'classes':1,
			'GridSize':GridSize,
			'layer_nums':   [2, 2, 4, 4, 6],
			'factor':2,
			'norm_method': 'layer',
			'pool_method':'max',
			'padding':'same',
			'padding_mode':'reflect',
			'end_padding':'same',
			'end_padding_mode':'reflect',
			'act':'relu'
		},
		log_dir=f"./all_logs",
		lr=1e-3,
		total_epochs=[150],
		tag=tag,
		loss_fn=mse_loss,
		model_save_path=f"./model_save",
		hyper_params_save_path=f"./hyper_parameters",
		)
	
	trainer.fit_loop()