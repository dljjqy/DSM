import sys
sys.path.append('../')

from BaseTrainer import BaseTrainer
from SegModel3D import CNN

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.lines import Line2D as mlines
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def force_3D(x, y, z, nu):
	return -(2*(-x*y*(x - 1)*(y - 1)*(2*nu*z**2 + nu*z*(z - 1)*(2*nu*z**2 + 1) + 2*nu*z*(z - 1) + 1) - x*z*(x - 1)*(z - 1)*(2*nu*y**2 + nu*y*(y - 1)*(2*nu*y**2 + 1) + 2*nu*y*(y - 1) + 1) - y*z*(y - 1)*(z - 1)*(2*nu*x**2 + nu*x*(x - 1)*(2*nu*x**2 + 1) + 2*nu*x*(x - 1) + 1))*np.exp(nu*(x**2 + y**2 + z**2)))

def sol_3D(x, y, z, nu):
	return x*(1-x)*y*(1-y)*z*(1-z)*np.exp(nu*(x**2+y**2+z**2))

def hard_encode(x, gd=0.0):
	return F.pad(x, (1, 1, 1, 1, 1, 1), 'constant', gd)

class ConvJac3D(torch.nn.Module):
	def __init__(self, dtype, device, h):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.h = h

		self.k = self._get_kernel([
			[[0, 0, 0], [0, 1, 0], [0, 0, 0]], 
			[[0, 1, 0], [1, 0, 1], [0, 1, 0]], 
			[[0, 0, 0], [0, 1, 0], [0, 0, 0]]
			])

	def _get_kernel(self, k):
		k = torch.tensor(k, requires_grad=False)
		k = k.view(1, 1, 3, 3, 3).repeat(1, 1, 1, 1, 1).to(self.dtype).to(self.device)
		return k
	
	def _step(self, u, f):
		with torch.no_grad():
			w = (F.conv3d(u, self.k)+  self.h**2 * f[..., 1:-1, 1:-1, 1:-1])/6.0
		return w
		
	def forward(self, pre, f, max_iter):
		with torch.no_grad():
			w = self._step(pre, f)
			for i in range(1, max_iter):
				w = self._step(hard_encode(w), f)
			return w 

class C1TrainDs(Dataset):
	def __init__(self, GridSize, trainN, dtype=torch.float, device='cuda'):
		super().__init__()
		self.GridSize = GridSize
		self.trainN = trainN
		self.dtype = dtype
		self.device = device

		self.nu = np.random.uniform(0.5, 1.5, trainN)
		self.xx, self.yy, self.zz = np.meshgrid(
			np.linspace(0, 1, GridSize),
			np.linspace(0, 1, GridSize),
			np.linspace(0, 1, GridSize)
		)
	
	def __len__(self):
		return self.trainN
	
	def __getitem__(self, idx):
		source = force_3D(self.xx, self.yy, self.zz, self.nu[idx])
		return torch.from_numpy(source)[None, ...].to(self.dtype).to(self.device)

class C1ValDs(C1TrainDs):
	def __init__(self, GridSize, trainN, dtype=torch.float, device='cuda'):
		super().__init__(GridSize, trainN, dtype, device)
		self.nu = np.linspace(0.5, 1.5, trainN)
		
	def __getitem__(self, idx):
		nu = self.nu[idx]
		source = force_3D(self.xx, self.yy, self.zz, nu)
		target = sol_3D(self.xx, self.yy, self.zz, nu)
		return torch.from_numpy(source)[None, ...].to(self.dtype).to(self.device), torch.from_numpy(target)[None, ...].to(self.dtype).to(self.device)

class Trainer3D(BaseTrainer):
	def __init__(self, max_iter, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.init_dl()
		self.gen_mesh()
		self.max_iter = max_iter
		self.generator = ConvJac3D(self.dtype, self.device, self.h)
		self.gd = 0.0
	
	def init_network(self, kwargs):
		net_kwargs = kwargs.copy()
		name = net_kwargs.pop('model_name')
		net = CNN(name, net_kwargs, self.dtype, self.device)
		self.net = net.to(self.dtype)

	def hard_encode(self, x):
		return torch.nn.ConstantPad3d((1, 1, 1, 1, 1, 1), x)

	def init_dl(self):
		train_ds = C1TrainDs(self.GridSize, self.trainN, self.dtype, self.device)
		self.train_dl = DataLoader(train_ds, self.batch_size, shuffle=False)

		val_ds = C1ValDs(self.GridSize, self.valN, self.dtype, self.device)
		self.val_dl = DataLoader(val_ds, self.batch_size, shuffle=False)

	@property
	def name(self):
		return f"{self.tag}-3D"
	
	def l2(self, pre, ans):
		diff = (pre - ans[..., 1:-1, 1:-1, 1:-1]) ** 2 * self.h**3
		l2_errors = torch.sqrt(torch.sum(diff, dim=(1, 2, 3, 4)))
		return l2_errors.mean() 

	def gen_mesh(self):
		self.xx, self.yy = np.meshgrid(
			np.linspace(0, 1, self.GridSize),
			np.linspace(0, 1, self.GridSize),
		)
		self.h = 1 / (self.GridSize - 1)

	def reboot(self):
		self.config_optimizer(self.lr)

	def hyper_param_need2save(self):
		param = {
			'GridSize': self.GridSize,
			'area': self.area,
			'trainN': self.trainN,
			'valN': self.valN,
			'Batch_size': self.batch_size,
			'lr':self.lr,
			'epochs':self.total_epochs,
			'tag':self.tag,
			'dtype':self.dtype_name,
			'device':self.device,
			'net_kwargs': self.net_kwargs,
		}
		return param
	
	def train_step(self, source):
		# Prediction			
		pre = self.net(source)
		with torch.no_grad():
			label = self.generator(hard_encode(pre), source, self.max_iter)
		loss = self.loss_fn(pre, label)
		
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		loss_val = loss.item()
		self.writer.add_scalar("Train-Loss", loss_val, self.train_global_idx)
		self.train_global_idx += 1
		return loss_val

	def train_loop(self):
		self.net.train()
		errors = []
		for source in tqdm(self.train_dl, desc='Training Loop:', leave=False, position=1):
			error = self.train_step(source)
			errors.append(error)		
		error = np.array(errors).mean()
		return error

	def val_step(self, source, target):
		pre = self.net(source)
		real_error = self.l2(pre, target).item()

		self.writer.add_scalar("Val-RealError", real_error, self.train_global_idx)
		self.val_global_idx += 1
		return real_error

	def val_loop(self):
		self.net.eval()
		for source, ref in tqdm(self.val_dl, desc='Validation Loop:', leave=False, position=2):
			val_real_loss = self.val_step(source, ref)
		return val_real_loss

	def fit_loop(self):
		for epoch_num in self.total_epochs:
			self.reboot()

			for _ in tqdm(range(epoch_num), desc='Training Epoch:', leave=True):
				self.epoch_reboot()

				new_train_error = self.train_loop()
				with torch.no_grad():
					new_val_real_loss = self.val_loop()

				self.lr_scheduler.step()
				self.global_epoch_idx += 1
				torch.save(self.net.state_dict(), f'{self.model_save_path}/last.pt')
				
				self.save_best_train_error(new_train_error)
				self.save_best_val_real(new_val_real_loss)
			
	def val_plot(self, source, pre, ans, name='Val', z=0.5):
		xx, yy = self.xx, self.yy
		z_index = int(z * self.GridSize)
		source = source.reshape(self.GridSize,self.GridSize, self.GridSize)[..., z_index]
		pre = pre.reshape(self.GridSize,self.GridSize, self.GridSize)[..., z_index]
		ans = ans.reshape(self.GridSize,self.GridSize, self.GridSize)[..., z_index]

		fig = plt.figure()
		fig.suptitle(name, fontsize=20)
		fig.set_figheight(20)
		fig.set_figwidth(20)

		ax1 = fig.add_subplot(2, 2, 1, aspect="equal")
		ax2 = fig.add_subplot(2, 2, 2, projection='3d')
		ax3 = fig.add_subplot(2, 2, 3, aspect="equal")
		ax4 = fig.add_subplot(2, 2, 4, aspect="equal")
		
		im = ax1.imshow(source, origin='lower')
		ax1.set_title(f'$Source$', fontsize=20)
		cbar=plt.colorbar(im, shrink=0.85, ax=ax1)
		cbar.ax.tick_params(labelsize=20)

		ax2.set_title(f'$Prediction$', fontsize=20)
		surf_pre = ax2.plot_surface(xx, yy, pre)
		cbar=plt.colorbar(surf_pre, shrink=0.85, ax=ax2)
		cbar.ax.tick_params(labelsize=20)

		diff = np.abs(pre - ans)
		ctf = ax3.contourf(xx, yy, diff, levels=50)
		ax3.set_title(f'$Difference$', fontsize=20)
		cbar=plt.colorbar(ctf, shrink=0.85, ax=ax3)
		cbar.ax.tick_params(labelsize=20)

		if levels is None:
			levels = np.linspace(ans.min(), ans.max(), 10)[2:-2] 
		ct1 = ax4.contour(xx, yy, pre, colors='r', linestyles='dashed', linewidths=1.5,  levels=levels)
		ct2 = ax4.contour(xx, yy, ans, colors='b', linestyles='solid', linewidths=2, levels=levels)
		ax4.clabel(ct1, inline=False, fontsize=20)
		ax4.clabel(ct2, inline=False, fontsize=20)
		blue_line = mlines.Line2D([], [], color='blue', markersize=20, label='ref')
		red_line = mlines.Line2D([], [], color='red', markersize=20, label='pre')
		ax4.legend(handles=[blue_line, red_line], fontsize=20 )

		fig.tight_layout()
		return fig

if __name__ == "__main__":
	from torch.nn.functional import mse_loss
	GridSize = 128
	tag = f"C1Jac5-128-float"
	trainer = Trainer3D(
		max_iter = 5,
		dtype='float',
		device="cuda",
		area=((0,0,0), (1,1,1)),
		GridSize=GridSize,
		trainN=15000,
		valN=100,
		batch_size=10,
		net_kwargs={
			'model_name': 'UNet',
			'Block': "ResBottleNeck",
			'planes':8,
			'in_channels':1,
			'classes':1,
			'GridSize':GridSize,
			'layer_nums':[2,4,4,6,6],
			'factor':2,
			'norm_method': 'layer',
			'pool_method':'max',
			'padding':'same',
			'padding_mode':'zeros',
			'end_padding':'valid',
			# 'end_padding_mode':'zeros',
			'act': 'tanh',
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