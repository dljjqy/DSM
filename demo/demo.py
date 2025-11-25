# Test all possible residual loss function
# All experiments have illustrated that the residual loss function does not work

import sys

import torch.utils
sys.path.append('../')

import numpy as np
import torch
import numpy as np

from BaseTrainer import BaseTrainer
from utils import *
from MyPlot import *
from torch.utils.data import Dataset

from tqdm import tqdm

def u0(x, y, k=8):
	return np.sin(k* np.pi * x)*np.sin(k * np.pi * y)

def f0(x, y, k=8):
	return 2 * k**2 * np.pi**2 * u0(x, y, k)

class C1DsOnlyQ(Dataset):
	def __init__(self, k, DataN, GridSize, area=((-1, -1), (1, 1)), dtype=torch.float, device='cuda', ):
		self.gen_mesh(GridSize, area)
		force = f0(self.xx, self.yy, k)[np.newaxis, np.newaxis, ...]
		ref = u0(self.xx, self.yy, k)[np.newaxis, np.newaxis, ...]
		self.DataN = DataN
		self.Q = np.random.uniform(1, 2, (DataN, 1, 1, 1))
		self.forces = torch.from_numpy(self.Q * force).to(device).to(dtype)
		self.data = torch.from_numpy(self.Q * force).to(device).to(dtype)
		self.refs = torch.from_numpy(self.Q * ref).to(device).to(dtype)
		
	def gen_mesh(self, GridSize, area):
		(self.left, self.bottom), (self.right, self.top) = area
		self.xx, self.yy = np.meshgrid(
			np.linspace(self.left, self.right, GridSize),
			np.linspace(self.bottom, self.top, GridSize),
		)
		self.h = (self.right - self.left) / (self.GridSize - 1)
	
	def __len__(self):
		return self.DataN
	
	def __getitem__(self, index):
		f = self.forces[index]
		x = self.data[index]
		return x, f

class C3DsOnlyQ(Dataset):
	def __init__(self, k, DataN, GridSize, area=((-1, -1), (1, 1)), dtype=torch.float, device='cuda', ):
		self.gen_mesh(GridSize, area)
		force = f0(self.xx, self.yy, k)
		ref = u0(self.xx, self.yy, k)[np.newaxis, np.newaxis, ...]
		self.DataN = DataN
		self.Q = np.random.uniform(1, 2, (DataN, 1, 1, 1))
		self.forces = torch.from_numpy(self.Q * force).to(device).to(dtype)
		data = []
		for q in self.Q:
			data.append(np.stack([self.xx, self.yy, q*force]))
		data = torch.from_numpy(np.stack(data)).to(device).to(dtype)
		self.data = torch.from_numpy(self.Q * force).to(device).to(dtype)
		self.refs = torch.from_numpy(self.Q * ref).to(device).to(dtype)
		
	def gen_mesh(self, GridSize, area):
		(self.left, self.bottom), (self.right, self.top) = area
		self.xx, self.yy = np.meshgrid(
			np.linspace(self.left, self.right, GridSize),
			np.linspace(self.bottom, self.top, GridSize),
		)
		self.h = (self.right - self.left) / (self.GridSize - 1)
	
	def __len__(self):
		return self.DataN
	
	def __getitem__(self, index):
		f = self.forces[index]
		x = self.data[index]
		return x, f

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

class ResMonitor(torch.nn.Module):
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
		res = F.conv2d(u, self.k) - f[..., 1:-1, 1:-1] * self.h**2
		
		res = hard_encode(res, self.gd)
		res = F.conv2d(res, self.k)
		return torch.abs(res)

class ResidualLoss_bias(torch.nn.Module):
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
		u = self.hard_encode(pre + self.h**2 * torch.rand_like(pre, requires_grad=False))
		y = F.conv2d(u, self.k) 
		return self.loss_fn(y.squeeze(), self.h**2 * f[..., 1:-1, 1:-1])
	
class ResidualLoss_mask(torch.nn.Module):
	def __init__(self, dtype, device, h, loss_fn, gd=0, p=0.001):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.h = h
		self.loss_fn = loss_fn
		self.gd = gd

		self.k = self._get_kernel([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
		self.hard_encode = lambda x: hard_encode(x, self.gd)
		self.p = p

	def _get_kernel(self, k):
		k = torch.tensor(k, requires_grad=False)
		k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(self.dtype).to(self.device)
		return k
		
	def forward(self, pre, f, ans):
		prob_tensor = torch.full(pre.shape, self.p)
		mask = torch.bernoulli(prob_tensor).to(self.dtype).to(self.device)
		
		u = self.hard_encode(pre)
		y = F.conv2d(u, self.k) 
		res_loss =  self.loss_fn(y.squeeze(), self.h**2 * f[..., 1:-1, 1:-1])

		real_loss = self.loss_fn(mask * u[..., 1:-1, 1:-1], mask * ans[..., 1:-1, 1:-1])
		return res_loss + real_loss

class ResidualLoss_maskpre(torch.nn.Module):
	def __init__(self, dtype, device, h, loss_fn, gd=0, p=0.0001):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.h = h
		self.loss_fn = loss_fn
		self.gd = gd

		self.k = self._get_kernel([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
		self.hard_encode = lambda x: hard_encode(x, self.gd)
		self.p = p

	def _get_kernel(self, k):
		k = torch.tensor(k, requires_grad=False)
		k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(self.dtype).to(self.device)
		return k
		
	def forward(self, pre, f, ans):
		u = self.hard_encode(pre)
		prob_tensor = torch.full(u.shape, self.p)
		mask = torch.bernoulli(prob_tensor).to(self.dtype).to(self.device)
		if mask.sum() == 0:
			mask[0, 0, 64, 64] += 1
		u = (1 - mask) * u + mask * ans
		y = F.conv2d(u, self.k) 
		res_loss =  self.loss_fn(y.squeeze(), self.h**2 * f[..., 1:-1, 1:-1])
		return res_loss 

class ResidualLoss_nograd(torch.nn.Module):
	def __init__(self, dtype, device, h, loss_fn, gd=0):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.h = h
		self.loss_fn = loss_fn
		self.gd = gd

		self.k1 = self._get_kernel([[0, -1, 0], [-1, 0, -1], [0, -1, 0]])
		self.k2 = self._get_kernel([[0, 0, 0], [0, 4, 0], [0, 0, 0]])
		
		self.hard_encode = lambda x: hard_encode(x, self.gd)

	def _get_kernel(self, k):
		k = torch.tensor(k, requires_grad=False)
		k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(self.dtype).to(self.device)
		return k
		
	def forward(self, pre, f, ans):
		u = self.hard_encode(pre)
		with torch.no_grad():
			y1 = self.h**2 * f[..., 1:-1, 1:-1] - F.conv2d(u, self.k1) 
		return self.loss_fn(F.conv2d(u, self.k2), y1)
	
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
		match self.method:
			case 'Res':
				self.loss = ResidualLoss(self.dtype, self.device, self.h, self.loss_fn)
			case 'Res_nograd':
				self.loss = ResidualLoss_nograd(self.dtype, self.device, self.h, self.loss_fn)
			case 'Res_mask':
				self.loss = ResidualLoss_mask(self.dtype, self.device, self.h, self.loss_fn)
			case 'Res_maskpre':
				self.loss = ResidualLoss_maskpre(self.dtype, self.device, self.h, self.loss_fn)
			
			case 'Data':
				self.loss = RealLoss(self.loss_fn)

			case _:
				maxiter = int(self.method[-1])
				self.loss = JacLoss(self.dtype, self.device, self.h, self.loss_fn, maxiter)
		
		self.monitor = ResMonitor(self.dtype, self.device, self.h, self.loss_fn)
		self.force = torch.from_numpy(f0(self.xx, self.yy, K)).to(self.dtype).to(self.device)
		self.ans = torch.from_numpy(u0(self.xx, self.yy, K)).to(self.dtype).to(self.device)
		self.data = torch.from_numpy(np.stack([self.xx, self.yy])).to(self.dtype).to(self.device)

	@property
	def name(self):
		return f"{self.tag}-{self.net.name()}-{self.GridSize}-{self.method}-{self.trainN}"
	
	def l2(self, pre, ans):
		diff = (pre - ans[..., 1:-1, 1:-1]) ** 2 * self.h**2
		l2_errors = torch.sqrt(torch.sum(diff, dim=(1, 2, 3)))
		# print(l2_errors.shape)

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
		Q = torch.from_numpy(np.random.uniform(1, 2, (self.batch_size, 1, 1, 1))).to(self.dtype).to(self.device)
		forces = torch.clone(torch.detach(Q * self.force[None, None, ...]))
		# data = torch.clone(torch.detach(forces))
		data = torch.clone(self.data).detach().repeat(self.batch_size, 1, 1, 1)
		refs = torch.clone(torch.detach(Q * self.ans[None, None, ...]))

		# refs, forces = [], []
		# for i in range(self.batch_size):
		# 	# k = np.random.choice([1, 2, 4, 8], 1)
		# 	k = 1
		# 	q = np.random.uniform(1, 2)
		# 	forces.append(q * f0(self.xx, self.yy, k)[np.newaxis, ...])
		# 	refs.append(q * u0(self.xx, self.yy, k)[np.newaxis, ...])
		
		# forces = torch.from_numpy(np.stack(forces)).to(self.dtype).to(self.device)
		# refs = torch.from_numpy(np.stack(refs)).to(self.dtype).to(self.device)
		# data = torch.clone(torch.detach(forces))
		
		pre = self.net(data)
		loss = self.loss(pre, forces, refs)
		
		# Backpropagation
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		with torch.no_grad():
			real_error = self.l2(pre, refs).item()
			loss_val = loss.item()
			self.writer.add_scalar("Train-Loss", loss_val, self.train_global_idx)
			self.writer.add_scalar("Train-RealError", real_error, self.train_global_idx)
			self.train_global_idx += 1

		return real_error

	def val_step(self, ):
		Q = torch.from_numpy(np.random.uniform(1, 2, (self.batch_size, 1, 1, 1))).to(self.dtype).to(self.device)
		forces = torch.clone(torch.detach(Q * self.force[None, None, ...]))
		# data = torch.clone(torch.detach(forces))
		data = torch.clone(self.data).detach().repeat(self.batch_size, 1, 1, 1)
		refs = torch.clone(torch.detach(Q * self.ans[None, None, ...]))
		
		# refs, forces = [], []
		# for i in range(self.batch_size):
		# 	# k = np.random.choice([1, 2, 4, 8], 1)
		# 	k = 1
		# 	q = np.random.uniform(1, 2)
		# 	forces.append(q * f0(self.xx, self.yy, k)[np.newaxis, ...])
		# 	refs.append(q * u0(self.xx, self.yy, k)[np.newaxis, ...])
		
		# forces = torch.from_numpy(np.stack(forces)).to(self.dtype).to(self.device)
		# refs = torch.from_numpy(np.stack(refs)).to(self.dtype).to(self.device)
		# data = torch.clone(torch.detach(forces))

		pre = self.net(data)
		loss_val = self.loss(pre, forces, refs).item()
		real_error = self.l2(pre, refs).item()

		self.writer.add_scalar(f"Val-Loss", loss_val, self.train_global_idx)
		self.writer.add_scalar("Val-RealError", real_error, self.train_global_idx)
		self.val_plot(forces[-1], pre[-1], refs[-1], k)

		self.val_global_idx += 1
		return loss_val, real_error

	def train_loop(self):
		self.net.train()
		
		errors = []
		for _ in tqdm(range(self.trainN), desc='Training Loop:', leave=False, position=1):
			error = self.train_step()
			errors.append(error)
		
		error = np.array(errors).mean()
		# self.lr_scheduler.step(error)
			
		return error

	def val_loop(self):
		self.net.eval()
		error,  val_real_loss = self.val_step()
		return val_real_loss, error

	def val_plot(self, force, pre, ans, k):
		pre = hard_encode(pre, 0).cpu().numpy().squeeze()

		# print(pre.shape)
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
	# from itertools import product
	GridSize = 256
	method = 'Res'
	k = 1
	tag = f"K={k}-{method}"
	trainer = Trainer(
		K=k,
		method=method,
		dtype='float',
		device="cuda",
		area=((-1, -1), (1, 1)),
		GridSize=GridSize,
		trainN=100,
		valN=100,
		batch_size=1,
		net_kwargs={
			'model_name': 'UNet',
			'Block': "ResBottleNeck",
			'planes':8,
			'in_channels':2,
			'classes':1,
			'GridSize':GridSize,
			'layer_nums':[2,2,2,2],
			'factor':2,
			'norm_method': 'batch',
			'pool_method':'max',
			'padding':'same',
			'padding_mode':'zeros',
			'end_padding':'valid',
			'end_padding_mode':'zeros',
			'act': 'tanh',
		},
		log_dir=f"./all_logs",
		lr=1e-3,
		total_epochs=[50],
		tag=tag,
		loss_fn=mse_loss,
		model_save_path=f"./model_save",
		hyper_params_save_path=f"./hyper_parameters",
		)
	
	trainer.fit_loop()