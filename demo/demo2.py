import sys
sys.path.append('../')

import torch
import numpy as np
import torch.nn.functional as F
from utils import *

from MyPlot import *
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from BaseTrainer import BaseTrainer
from itertools import product
from scipy.sparse.linalg import spsolve
from demo1 import f0


def hard_encode(x, gd):
	y = F.pad(x, (1, 1, 1, 1), "constant", value=gd)
	return y

class TrainDs(Dataset):
	def __init__(self, GridSize, dtype, device, trainN, area=((0, 0), (1, 1))):
		self.GridSize = GridSize
		self.dtype = dtype
		self.device = device
		(left, bottom), (right, top) = area
		self.h = (right - left) / (GridSize - 1)
		xx, yy = np.meshgrid(
			np.linspace(left, right, GridSize), np.linspace(bottom, top, GridSize)
		)
		self.force = normal(xx, yy, self.h, [(right-left)/2, (top - bottom)/2])
		# self.force = f0(xx, yy, k=4)
		# self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)
		self.area = area
		self.trainN = trainN
		
	def __len__(self):
		return self.trainN

	def __getitem__(self, index):
		force = torch.from_numpy(self.force).to(self.dtype).to(self.device)[None, ...]
		
		kappa = np.random.uniform(0.1, 2, (self.GridSize, self.GridSize))
		kappa = torch.from_numpy(kappa).to(self.dtype).to(self.device)[None, ...]
		data = torch.clone(torch.detach(kappa))
		return data, force, kappa
	
class ValDs(TrainDs):
	def __getitem__(self, index):
		kappa = np.random.uniform(0.1, 2, (self.GridSize, self.GridSize))
		A = reaction_A(self.GridSize, kappa).tocsr()
		b = reaction_b_dir(self.force, 0, self.h)
		ans = spsolve(A, b).reshape(self.GridSize, self.GridSize)
		
		force = torch.from_numpy(self.force).to(self.dtype).to(self.device)[None, ...]
		
		kappa = torch.from_numpy(kappa).to(self.dtype).to(self.device)[None, ...]
		data = torch.clone(torch.detach(kappa))
		ans = torch.from_numpy(ans).to(self.dtype).to(self.device)[None, ...]

		return data, force, kappa, ans
		
class JacGenerator(torch.nn.Module):
	def __init__(
		self, batch_size, GridSize, dtype, device, maxiter, area, gd=0
	):
		super().__init__()
		self.batch_size = batch_size
		self.GridSize = GridSize
		(left, bottom), (right, top) = area
		self.h = (right - left) / (GridSize - 1)
		self.dtype = dtype
		self.device = device
		self.maxiter = maxiter
		self.gd = gd
		self.hard_encode = lambda x: hard_encode(x, self.gd)

		self.k1 = self._get_kernel([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
		self.k2 = self._get_kernel([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
		self.k3 = self._get_kernel([[0, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 0]])

	def _get_kernel(self, k):
		k = torch.tensor(k, requires_grad=False)
		k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(self.dtype).to(self.device)
		return k

	def jac_step(self, pre, f, w):
		u = self.hard_encode(pre)

		force = f[..., 1:-1, 1:-1] * self.h**2
		y1 = F.conv2d(u, self.k1) * w[..., 1:-1, 1:-1]
		y2 = F.conv2d(w * u, self.k2)
		y3 = F.conv2d(w, self.k3)
		return (force + y1 + y2) / y3

	def forward(self, pre, f, kappa):
		with torch.no_grad():
			u = torch.clone(torch.detach(pre))
			f = torch.clone(torch.detach(f))
			kappa = torch.clone(torch.detach(kappa))
			for _ in range(self.maxiter):
				u = self.jac_step(u, f, kappa)
		return u

class DescentGenerator(JacGenerator):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def Ax(self, x, w):
		with torch.no_grad():
			y1 = F.conv2d(x, self.k1) * w[..., 1:-1, 1:-1]
			y2 = F.conv2d(w * x, self.k2)
			y3 = F.conv2d(w, self.k3)
			y = y3 * x[..., 1:-1, 1:-1] - y1 - y2
			return y

	def step(self, pre, f, w):
		x = self.hard_encode(pre)
		b = f[..., 1:-1, 1:-1] * self.h**2
		r = b - self.Ax(x, w)

		Ar = self.Ax(self.hard_encode(r), w)
		alpha = torch.sum(r**2, dim=(1, 2, 3), keepdim=True) / torch.sum(r * Ar, dim=(1, 2, 3), keepdim=True)
		# print(alpha.shape)
		y = pre + alpha * r
		return y

	def forward(self, pre, f, kappa):
		with torch.no_grad():
			u = torch.clone(torch.detach(pre))
			f = torch.clone(torch.detach(f))
			kappa = torch.clone(torch.detach(kappa))
			for i in range(self.maxiter):
				u = self.step(u, f, kappa)
		return u

class MatDescentGenerator(JacGenerator):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def get_Ab(self, force, kappas):
		A, B = [], []
		for i in range(self.batch_size):
			k = kappas[i].cpu().numpy().squeeze()
			f = force[i].cpu().numpy().squeeze()
			A.append(
				coo2tensor(reaction_A(self.GridSize, k).tocoo(), self.device, self.dtype)
				)
			B.append(
				torch.from_numpy(reaction_b_dir(f, 0.0, self.h))
				)
		A = torch.stack(A)
		B = torch.stack(B).to(self.device).to(self.dtype)
		return A, B
	
	def step(A, B, u):
		r = B - torch.bmm(A, u)
		alpha = bvi(r, r) / bvi(r, torch.bmm(A, r))
		new_u = u + alpha * r
		return new_u
	
	def forward(self, pre, f, kappa):
		with torch.no_grad():
			u = torch.clone(torch.detach(pre)).flatten(1,-1)
			# u = hard_encode(u, 0.0).flatten(1)
			f = torch.clone(torch.detach(f))
			kappa = torch.clone(torch.detach(kappa))

			A, B = self.get_Ab(f, kappa)
			for i in range(self.maxiter):
				u = self.step(A, B, u)
		return u.reshape(self.batch_size, 1, self.GridSize, self.GridSize)

		
class Trainer(BaseTrainer):
	def __init__(
		self,
		gd=0,
		method = 'Jac-3',
		*args,
		**kwargs,
		):
		self.gd = gd
		self.method = method
		super().__init__(*args, **kwargs)
		self.h = 1.0 / (self.GridSize - 1)
		self.generator = self.init_generator(method)

	@property
	def name(self):
		return f"{self.tag}-{self.net.name}-{self.GridSize}-{self.method}-{self.trainN}"

	def hyper_param_need2save(self):	
		param = {
			"gd": self.gd,
			"GridSize": self.GridSize,
			"area": self.area,
			"trainN": self.trainN,
			"valN": self.valN,
			"Batch_size": self.batch_size,
			"lr": self.lr,
			"epochs": self.total_epochs,
			"tag": self.tag,
			"net_kwargs": self.net_kwargs,
		}
		return param

	def init_traindl(self):
		train_ds = TrainDs(self.GridSize, self.dtype, self.device, self.trainN, self.area)
		self.train_dl = DataLoader(train_ds, self.batch_size, shuffle=True)

	def init_valdl(self):
		val_ds = ValDs(self.GridSize, self.dtype, self.device, self.valN, self.area)
		self.val_dl = DataLoader(val_ds, 1, shuffle=False)

	def reboot(self):
		self.init_traindl()
		self.init_valdl()
		self.config_optimizer(self.lr)
		# self.init_generator(self.method)

	def epoch_reboot(self):
		pass

	def init_generator(self, method):
		generator_method, maxiter = method.split('-')
		maxiter = int(maxiter)
		match generator_method:
			case 'Jac':
				generator = JacGenerator(
					self.batch_size,
					self.GridSize,
					self.dtype,
					self.device,
					maxiter,
					self.area,
					self.gd,
				)
			case 'Desc':
				generator = DescentGenerator(
					self.batch_size,
					self.GridSize,
					self.dtype,
					self.device,
					maxiter,
					self.area,
					self.gd,
				)
			case 'MatDesc':
				generator = MatDescentGenerator(
					self.batch_size,
					self.GridSize,
					self.dtype,
					self.device,
					maxiter,
					self.area,
					self.gd,
				)
		return generator
	
	def train_step(self, x, f, kappa):
		pre = self.net(x)
		with torch.no_grad():
			label = self.generator(torch.clone(torch.detach(pre)), f, kappa)
			# print((label == pre).all())

		train_loss = self.loss_fn(pre, label)

		self.optimizer.zero_grad()
		train_loss.backward()
		self.optimizer.step()

		return train_loss.item()

	def train_loop(self):
		self.net.train()
		for x, f, kappa in tqdm(
			self.train_dl, position=1, leave=True, desc="Training Loop:"
		):
			loss_val = self.train_step(x, f, kappa)
			self.train_global_idx += 1
			self.save_best_train_error(loss_val)
			self.writer.add_scalar("Train-Loss", loss_val, self.train_global_idx)

	def val_step(self, x, f, kappa, ans):
		pre = self.net(x)
		real_loss = self.loss_fn(hard_encode(pre, self.gd), ans)

		self.val_global_idx += 1
		self.writer.add_scalar("Val-RealLoss", real_loss.item(), self.val_global_idx)
		return pre, real_loss.item()

	def val_loop(self):
		self.net.eval()

		with torch.no_grad():
			for x, f, kappa, ans in self.val_dl:
				pre, real_loss = self.val_step(x, f, kappa, ans)
				self.val_global_idx += 1

				self.save_best_val_real(real_loss)

			self.val_plot(
				pre=hard_encode(pre, self.gd),
				kappa=kappa,
				ans=ans,
			)
			# self.writer.add_figure("ValFigure", fig, self.val_global_idx)
			
	def val_plot(self, pre, kappa, ans):
		pre = pre.cpu().numpy().squeeze()
		ans = ans.cpu().numpy().squeeze()
		kappa = kappa.cpu().numpy().squeeze()

		fig = multi_cof_draw_img(f"ValFigure", pre, ans, kappa, self.GridSize, a=1,)
		self.writer.add_figure(f"ValFigure", fig, self.val_global_idx)

		
	def fit_loop(self):
		self.reboot()
		for _ in tqdm(range(self.total_epochs[0]), desc='Training Epoch:', leave=True):
			self.epoch_reboot()
			self.train_loop()
			with torch.no_grad():
				self.val_loop()

			self.lr_scheduler.step()
			self.global_epoch_idx += 1
			torch.save(self.net.state_dict(), f'{self.model_save_path}/last.pt')


if __name__ == "__main__":
	from utils import set_seed

	set_seed(0)
	GridSize = 256
	tag = "Demo2"
	for trainN, k in product([100, 500, 5000], [1, 5, 25]):
		trainer = Trainer(
			gd=0,
			method=f'Jac-{k}',
			dtype=torch.float,
			device="cuda",
			area=((0, 0), (1, 1)),
			GridSize=GridSize,
			trainN=trainN,
			valN=10,
			batch_size=5,
			net_kwargs={
				"model_name": "segmodel",
				"Block": "ResBottleNeck",
				"planes": 6,
				"in_channels": 1,
				"classes": 1,
				"GridSize": GridSize,
				"layer_nums": [2, 2, 2, 2],
				"adaptor_nums": [2, 2, 2, 2],
				"factor": 2,
				"norm_method": "layer",
				"pool_method": "max",
				"padding": "same",
				"padding_mode": "zeros",
				"end_padding_mode": "zeros",
				"end_padding": "valid",
				"act": "tanh"
			},
			log_dir=f"./all_logs",
			lr=1e-3,
			total_epochs=[150],
			tag=tag,
			loss_fn=F.mse_loss,
			model_save_path=f"./model_save",
			hyper_params_save_path=f"./hyper_parameters",
		)
		trainer.fit_loop()
