import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad as pad

from utils import mmbv, bvi, hard_encode, kappa

def batchediv2tensor(indices, values, GridSize, dtype, device):
	batch_size ,_, N = indices.shape
	new_indices = []
	for k in range(batch_size):
		idx = torch.ones(1, N) * k
		new_indices.append(torch.concatenate([idx, indices[k]], dim=0))
	new_indices = torch.concatenate(new_indices, dim=1)
	new_values = torch.flatten(values)

	BatchedA = torch.sparse_coo_tensor(
        new_indices, new_values, (batch_size, GridSize**2, GridSize**2), 
        device=device, requires_grad=False, dtype=dtype, is_coalesced=True)

	return BatchedA	

class BatchedL2(nn.Module):
	def __init__(self, h):
		super().__init__()
		self.h = h

	def forward(self, pre, ref):
		diff = (pre - ref)**2 * self.h**2
		l2_errors = torch.sqrt(torch.sum(diff, dim=0))
		return l2_errors.mean()

class PinnGenerator(torch.nn.Module):
	def __init__(self, GridSize, device, maxiter, area, prev_net=None, gd=0):
		super().__init__()
		self.GridSize = GridSize
		(left, bottom), (right, top) = area
		self.h = (right - left) / (GridSize - 1)
		self.device = device
		self.maxiter = maxiter
		self.gd = gd

		self.k1 = self._get_kernel([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
		self.k2 = self._get_kernel([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
		self.k3 = self._get_kernel([[0, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 0]])

		self.prev_net = prev_net

		self.hard_encode = lambda x: hard_encode(x, self.gd)

	def _get_kernel(self, k):
		k = torch.tensor(k, requires_grad=True)
		k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).float().to(self.device)
		return k

	def jac_step(self, x, pre, f, mu):
		u = self.hard_encode(pre)
		if self.prev_net is None:
			w = torch.ones_like(f, requires_grad=False)
		else:
			w = self.hard_encode(self.prev_net(x))
		
		w = kappa(w, mu)
		force = f[..., 1:-1, 1:-1] * self.h**2
		y1 = F.conv2d(u, self.k1) * w[..., 1: -1, 1: -1]        
		y2 = F.conv2d(w * u, self.k2)
		y3 = F.conv2d(w, self.k3)
		return (force + y1 + y2) / y3
	
	def forward(self, x, pre, f, mu):
		with torch.no_grad():
			y = self.jac_step(x, pre, f, mu)
			for _ in range(self.maxiter):
				y = self.jac_step(x, y, f, mu)
		return y


class JacBatched(nn.Module):
	def __init__(self, A,  dtype, device):
		super().__init__()
		with torch.no_grad():  
			A = A.coalesce()
			idx = A.indices()
			mask = (idx[1] == idx[2])

			D_idx = idx[:, mask]
			D_val = A.values()[mask]

			D = torch.sparse_coo_tensor(
				indices=D_idx, 
				values=D_val,
				dtype=dtype,
				device=device,
				requires_grad=False
				)
			self.M = A - D
			del D

			self.invD = torch.sparse_coo_tensor(
				indices=D_idx, 
				values=1.0 / D_val,
				dtype=dtype,
				device=device,
				requires_grad=False
				) 
			
			self.dtype = dtype
			self.device = device

	def forward(self, u, b, maxiter):
		'''
		x: torch.Tensor with shape B x 1 x N x N
		b: torch.Tensor with shape B x N**2
		'''
		original_shape = u.shape
		with torch.no_grad():
			x = torch.flatten(u, 1, -1)[..., None]
			for _ in range(maxiter):
				Mx = torch.bmm(self.M, x)
				x = torch.bmm(self.invD, (b - Mx)) 
			x_new = x.reshape(original_shape)
			return x_new.to(self.dtype).to(self.device)
		
class JacTorch(nn.Module):
	def __init__(self, A, device, dtype):
		super().__init__()
		with torch.no_grad():  
			A = A.coalesce()
			idx = A.indices()
			mask = (idx[-2] == idx[-1])

			D_idx = idx[:, mask]
			D_val = A.values()[mask]

			D = torch.sparse_coo_tensor(
				indices=D_idx, 
				values=D_val,
				dtype=dtype,
				device=device,
				requires_grad=False
				)
			self.M = A - D
			del D

			self.invD = torch.sparse_coo_tensor(
				indices=D_idx, 
				values=1.0 / D_val,
				dtype=dtype,
				device=device,
				requires_grad=False
				) 
			
			self.dtype = dtype
			self.device = device

	def forward(self, u, b, maxiter):
		'''
		x: torch.Tensor with shape B x 1 x N x N
		b: torch.Tensor with shape B x N**2
		'''
		original_shape = u.shape
		with torch.no_grad():
			x = torch.flatten(u, 1, -1)
			for _ in range(maxiter):
				Mx = mmbv(self.M, x)
				x = mmbv(self.invD, (b - Mx)) 
			x_new = x.reshape(original_shape)
			return x_new.to(self.dtype).to(self.device)

class CGBatched(nn.Module):
	def __init__(self, A, device, dtype):
		super().__init__()
		with torch.no_grad():
			self.dtype = dtype
			self.device = device
			self.A = A.to(dtype).to(device)

	def forward(self, u, b, maxiter):
		with torch.no_grad():
			original_shape = u.shape
			x = torch.flatten(u, 1, -1)[..., None]
			y = self.rhs_cg(x, b, maxiter)

		return y.reshape(original_shape)

	def rhs_cg(self, x, b, max_iters=20):
		r = b - torch.bmm(self.A, x)
		p = r
		for _ in range(max_iters):
			rr = bvi(r, r)[..., None]
			
			Ap = torch.bmm(self.A, p)
			
			alpha = rr / bvi(p, Ap)[..., None]
			x = x + alpha * p
			r1 = r - alpha * Ap
			beta = bvi(r1, r1)[..., None] / rr
			p = r1 + beta * p
			r = r1
		return x


class CGTorch(nn.Module):
	def __init__(self, A, device, dtype):
		super().__init__()
		with torch.no_grad():
			self.dtype = dtype
			self.device = device
			self.A = A.to(dtype).to(device)

	def forward(self, u, b, maxiter):
		with torch.no_grad():
			original_shape = u.shape
			x = torch.flatten(u, 1, -1)

			y = self.rhs_cg(x, b, maxiter)
		return y.reshape(original_shape)

	def rhs_cg(self, x, b, max_iters=20):
		r = b - mmbv(self.A, x)
		p = r
		for _ in range(max_iters):
			rr = bvi(r, r)
			Ap = mmbv(self.A, p)
			alpha = rr / bvi(p, Ap)
			x = x + alpha * p
			r1 = r - alpha * Ap
			beta = bvi(r1, r1) / rr
			p = r1 + beta * p
			r = r1
			print(f"error: {r.mean().item():.3e}")
		return x

class ConvJac(nn.Module):
	def __init__(self, dtype, device, GridSize, h):
		super().__init__()
		self.dtype = dtype
		self.device = device
		self.GridSize = GridSize
		self.h = h

		self.gd = 0
		self.gn = 1
	
	def forward(self, max_iter, u, b, K):
		with torch.no_grad():
			left, right, up, low, diag = self.compute_A(K)
			y = self.jac_step(u, b, left, right, up, low, diag)
			for _ in range(1, max_iter):
				y = self.jac_step(y, b, left, right, up, low, diag)
			return y

	def compute_A(self, K):
		'''Only up boundary is Dirichlet boundary with gd=0, others are Neumann boundary.
		The flux over the left and right are gn=0, and the flux over the low side is gn=1.0.
		'''
		lbd = 1.0/K
		left = 2 / (lbd[..., :, 0:-1] + lbd[..., :, 1:])
		left = pad(left, (1, 0, 0, 0), 'constant')

		right = 2 / (lbd[..., :, 0:-1,] + lbd[..., :, 1:])
		right = pad(right, (0, 1, 0, 0), 'constant')

		top_lbd = pad(lbd, (0, 0, 0, 1), 'constant')
		up = 2 / (top_lbd[...,0:-1, :] + top_lbd[...,1:, :])

		low_flux = 2 / (lbd[...,0:-1,:] + lbd[...,1:,:])
		low = pad(low_flux, (0,0,1,0), 'constant')


		diag = left + right + up + low
		return left, right, up, low, diag

	def jac_step(self, u, b, left, right, up, low, diag):
		with torch.no_grad():
			left_flux = -left * pad(u, (1,0,0,0), 'constant', 0.0)[..., :, :-1]
			right_flux = -right * pad(u, (0,1,0,0), 'constant', 0.0)[..., :, 1:]
			low_flux =  -low * pad(u, (0,0,1,0), 'constant', 0.0)[..., :-1, :]

			up_flux = -up * (pad(u, (0,0,0,1),'constant', self.gd)[..., 1:, :])
			
			# Jacobi
			pseudo_label = (b - (left_flux + right_flux + up_flux + low_flux)) / diag 
			return pseudo_label 