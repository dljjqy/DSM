# Ablation study operated on the PDE without parameters
# It is used to test the performance of the UNet with different optimization methods, activation functions and the loss function.

import sys

import torch.utils
sys.path.append('../')

import numpy as np
import torch
import numpy as np

from BaseTrainer import BaseTrainer
from utils import *
from MyPlot import *
from torch.autograd import grad

from tqdm import tqdm

def u0(x, y, k=8):
	return np.sin(k* np.pi * x)*np.sin(k * np.pi * y)

def f0(x, y, k=8):
	return 2 * k**2 * np.pi**2 * u0(x, y, k)

class ResidualLoss(torch.nn.Module):
	def __init__(self, dtype, device, h, loss_fn, gd=0):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.h = h
		self.loss_fn = loss_fn
		self.gd = gd

		self.k = self._get_kernel([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
		self.hard_encode = lambda x: hard_encode(x, self.gd)

	def _get_kernel(self, k):
		k = torch.tensor(k, requires_grad=False)
		k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(self.dtype).to(self.device)
		return k
		
	def forward(self, pre, f, ans):
		u = self.hard_encode(pre)
		y = F.conv2d(u, self.k) 
		return self.loss_fn(y, self.h**2 * f[..., 1:-1, 1:-1])
	
class MatrixResidualLoss(torch.nn.Module):
	def __init__(self, dtype, device, GridSize, h, loss_fn, maxiter=5, gd=0):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.h = h
		self.loss_fn = loss_fn
		self.maxiter = maxiter
		self.gd = gd

		A = reaction_A(GridSize, a=np.ones((GridSize**2, GridSize**2))).tocoo()
		self.A = coo2tensor(A, device, dtype)
		self.hard_encode = lambda x: hard_encode(x, self.gd)
		
	def forward(self, pre, f, ans):
		u = self.hard_encode(pre).reshape(-1)
		with torch.no_grad():
			f = f.cpu().numpy().squeeze()
			b = reaction_b_dir(f, 0, self.h)
			b = torch.from_numpy(b).to(self.device).to(self.dtype)
		# r = (self.A @ u - b).squeeze()
		# return self.loss_fn(r, torch.zeros_like(r))
		return self.loss_fn(self.A@u, b)
	
class EnergyResidualLoss(torch.nn.Module):
	def __init__(self, dtype, device, GridSize, h, loss_fn, maxiter=5, gd=0):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.h = h
		self.loss_fn = loss_fn
		self.maxiter = maxiter
		self.gd = gd

		A = reaction_A(GridSize, a=np.ones((GridSize**2, GridSize**2))).tocoo()
		self.A = coo2tensor(A, device, dtype)
		self.hard_encode = lambda x: hard_encode(x, self.gd)
		
	def forward(self, pre, f, ans):
		u = self.hard_encode(pre).reshape(-1)
		with torch.no_grad():
			f = f.cpu().numpy().squeeze()
			b = reaction_b_dir(f, 0, self.h)
			b = torch.from_numpy(b).to(self.device).to(self.dtype)
		
		loss = 0.5 *torch.dot(u, self.A @ u) - torch.dot(b, u)
		return loss
	
class DescentLoss(torch.nn.Module):
	def __init__(self, dtype, device, GridSize, h, loss_fn, maxiter=5, gd=0):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.h = h
		self.loss_fn = loss_fn
		self.maxiter = maxiter
		self.gd = gd

		A = reaction_A(GridSize, a=np.ones((GridSize**2, GridSize**2))).tocoo()
		self.A = coo2tensor(A, device, dtype)
		self.hard_encode = lambda x: hard_encode(x, self.gd)
		
	def forward(self, pre, f, ans):
		u = self.hard_encode(pre).reshape(-1)
		with torch.no_grad():
			f = f.cpu().numpy().squeeze()
			b = reaction_b_dir(f, 0, self.h)
			b = torch.from_numpy(b).to(self.device).to(self.dtype)

			x = torch.clone(torch.detach(u))
			for i in range(self.maxiter):
				r = (b - self.A@x).squeeze()
				alpha = torch.dot(r, r) / torch.dot(r, self.A@r)
				x = x + alpha * r
		loss = self.loss_fn(u, x)
		return loss
		
class JacLoss(torch.nn.Module):
	def __init__(self, dtype, device, h, loss_fn, maxiter=1, gd=0):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.h = h
		self.loss_fn = loss_fn
		self.gd = gd
		self.maxiter = maxiter

		self.k = self._get_kernel([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
		self.hard_encode = lambda x: hard_encode(x, self.gd)

	def _get_kernel(self, k):
		k = torch.tensor(k, requires_grad=False)
		k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(self.dtype).to(self.device)
		return k
	
	def _step(self, pre, f):
		with torch.no_grad():
			u = self.hard_encode(pre)
			w = F.conv2d(u, self.k) + 0.25 * self.h**2 * f[..., 1:-1, 1:-1]
		return w
		
	def forward(self, pre, f, ans):
		with torch.no_grad():
			w = self._step(pre, f)
			for i in range(1, self.maxiter):
				w = self._step(w, f)
		return self.loss_fn(pre, w)

class RealLoss(torch.nn.Module):
	def __init__(self, loss_fn):
		super().__init__()
		self.loss_fn = loss_fn

	def forward(self, pre, force, ans):
		return self.loss_fn(hard_encode(pre, 0.0).squeeze(), ans)

class Trainer(BaseTrainer):
	def __init__(
		self, 
		K,
		method, 
		*args, **kwargs
	):
		self.k = K
		self.method = method
		super().__init__(*args, **kwargs)

		self.gen_mesh()
		match self.method.split('-')[0]:
			case 'ConvRes':
				self.loss = ResidualLoss(self.dtype, self.device, self.h, self.loss_fn)
			case 'MatRes':
				self.loss = MatrixResidualLoss(self.dtype, self.device, self.GridSize, self.h, self.loss_fn)
			case 'EnergyRes':
				self.loss = EnergyResidualLoss(self.dtype, self.device, self.GridSize, self.h, self.loss_fn)
			case 'Data':
				self.loss = RealLoss(self.loss_fn)
			case 'Jac':
				maxiter = int(self.method[-1])
				self.loss = JacLoss(self.dtype, self.device, self.h, self.loss_fn, maxiter)
			case 'Desc':
				maxiter = int(self.method[-1])
				self.loss = DescentLoss(self.dtype, self.device, self.GridSize, self.h, self.loss_fn, maxiter, 0)

		self.force = torch.from_numpy(f0(self.xx, self.yy, K)).to(self.dtype).to(self.device)
		self.ans = torch.from_numpy(u0(self.xx, self.yy, K)).to(self.dtype).to(self.device)
		# self.monitor = ResidualLoss(self.dtype, self.device, self.h, self.loss_fn, 0.0)

	@property
	def name(self):
		return f"{self.tag}-{self.net.name}-{self.GridSize}-{self.method}-{self.trainN}"
	
	def l2(self, pre, ans):
		diff = (pre - ans[..., 1:-1, 1:-1]) ** 2 * self.h**2
		l2_errors = torch.sqrt(torch.sum(diff, dim=(1, 2, 3)))
		return l2_errors.mean() 

	def gen_mesh(self):
		(self.left, self.bottom), (self.right, self.top) = self.area
		self.xx, self.yy = np.meshgrid(
			np.linspace(self.left, self.right, self.GridSize),
			np.linspace(self.bottom, self.top, self.GridSize),
		)
		self.h = (self.right - self.left) / (self.GridSize - 1)

	def reboot(self):
		self.config_optimizer(self.lr)

	def hyper_param_need2save(self):
		param = {
			'GridSize': self.GridSize,
			'method': self.method,
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

	def config_optimizer(self, lr):
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
		self.optimizer.zero_grad()
		self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
		# self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5 )

	def train_step(self):
		# Prediction
		data = torch.from_numpy(np.stack([self.xx, self.yy])).to(self.device).to(self.dtype)
		data = data[None, ...]
		pre = self.net(data)
		
		force = torch.clone(torch.detach(self.force))[None, None, ...]
		ref = torch.clone(torch.detach(self.ans))
		loss = self.loss(pre, force, ref)
		
		# Backpropagation
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		with torch.no_grad():
			real_error = self.l2(pre, ref).item()
			loss_val = loss.item()
			self.writer.add_scalar("Train-Loss", loss_val, self.train_global_idx)
			self.writer.add_scalar("Train-RealError", real_error, self.train_global_idx)
			self.train_global_idx += 1

		return real_error

	def val_step(self, ):
		data = torch.from_numpy(np.stack([self.xx, self.yy])).to(self.device).to(self.dtype)
		data = data[None, ...]
		pre = self.net(data)
		
		force = torch.clone(torch.detach(self.force))[None, None, ...]
		ref = torch.clone(torch.detach(self.ans))
		loss_val = self.loss(pre, force, ref).item()
		real_error = self.l2(pre, ref).item()

		self.writer.add_scalar(f"Val-Loss", loss_val, self.train_global_idx)
		self.writer.add_scalar("Val-RealError", real_error, self.train_global_idx)
		self.val_plot(force.squeeze(), pre.squeeze(), ref.squeeze(), k)

		self.val_global_idx += 1
		return loss_val, real_error

	def train_loop(self):
		self.net.train()
		errors = []
		for _ in tqdm(range(self.trainN), desc='Training Loop:', leave=False, position=1):
			error = self.train_step()
			errors.append(error)
		
		error = np.array(errors).mean()
		return error

	def val_loop(self):
		self.net.eval()
		error,  val_real_loss = self.val_step()
		return val_real_loss, error

	def val_plot(self, force, pre, ans, k):
		pre = hard_encode(pre, 0).cpu().numpy().squeeze()
		ans = ans.cpu().numpy().squeeze()
		force = force.cpu().numpy().squeeze()

		fig = multi_cof_draw_img(f"K={k}", pre, ans, force, self.GridSize, a=1,)
		self.writer.add_figure(f"ValFigure", fig, self.val_global_idx)

	def fit_loop(self):
		for epoch_num in self.total_epochs:
			self.reboot()

			for _ in tqdm(range(epoch_num), desc='Training Epoch:', leave=True):
				self.epoch_reboot()

				new_train_error = self.train_loop()
				with torch.no_grad():
					new_val_real_loss, new_val_error = self.val_loop()

				self.lr_scheduler.step()
				self.global_epoch_idx += 1
				torch.save(self.net.state_dict(), f'{self.model_save_path}/last.pt')
				
				self.save_best_train_error(new_train_error)
				self.save_best_val_real(new_val_real_loss)
				self.save_best_val_error(new_val_error)

if __name__ == "__main__":
	from torch.nn.functional import mse_loss
	from itertools import product
	torch.manual_seed(0)

	GridSize = 256
	method = 'MatRes'
	act = 'tanh'
	# k = 2
	layer_nums = [2, 2, 2, 2]
	
	# for layer_nums in [
	# 	# [2, 2],
	# 	# [2, 2, 2],
	# 	[2, 2, 2, 2],
	# 	# [2, 2, 2, 2, 2],
	# ]:
	for k in [8]:
	# for method, act, norm in product(['Jac-3', 'Desc-3', 'EnergyRes', 'MatRes' ], ['relu', 'tanh'], ['batch', 'layer']):
	# for method, act in product([ 'EnergyRes' ], ['relu']):
		tag = f"k={k}"
		trainer = Trainer(
			K=k,
			method=method,
			dtype=torch.float,
			device="cuda",
			area=((-1, -1), (1, 1)),
			GridSize=GridSize,
			trainN=100,
			valN=1,
			batch_size=1,
			net_kwargs=
			{
				'model_name': 'segmodel',
				"Block": 'ResBottleNeck',
				"planes": 8,
				"in_channels": 2,
				"classes": 1,
				"GridSize": GridSize,
				"layer_nums": layer_nums,
				"adaptor_nums": layer_nums,
				"factor": 2,
				"norm_method": 'layer',
				"pool_method": "max",
				"padding": "same",
				"padding_mode": "zeros",
				"end_padding":"valid",
				"end_padding_mode": "zeros",
				"act": act
			},
			log_dir=f"./all_logs",
			lr=1e-2,
			total_epochs=[150],
			tag=tag,
			loss_fn=mse_loss,
			model_save_path=f"./model_save",
			hyper_params_save_path=f"./hyper_parameters",
			)
		
		trainer.fit_loop()