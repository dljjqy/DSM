import torch
import numpy as np
import torch.nn as nn
from utils import mmbv, bvi

class JacFixedB(nn.Module):
    def __init__(self, A, b, device, dtype):
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
            
            if isinstance(b, np.array):
                self.b = torch.from_numpy(b)[..., None].to(dtype).to(device)
            else:
                self.b = b[..., None].to(dtype).to(device)
            
            self.dtype = dtype
            self.device = device

    def forward(self, u, maxiter):
        '''
        x: torch.Tensor with shape B x 1 x N x N
        b: torch.Tensor with shape B x N**2
        '''
        original_shape = u.shape
        with torch.no_grad():
            x = torch.flatten(u, 1, -1)[..., None]
            for _ in range(maxiter):
                Mx = mmbv(self.M, x)
                x = mmbv(self.invD, (self.b - Mx)) 
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
            x = torch.flatten(u, 1, -1)[..., None]
            b = b[..., None]
            for _ in range(maxiter):
                Mx = mmbv(self.M, x)
                x = mmbv(self.invD, (b - Mx)) 
            x_new = x.reshape(original_shape)
            return x_new.to(self.dtype).to(self.device)

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


# class NpSolver:
#     def __init__(self, A, method):
#         self.A = A
#         match method:
#             case 'gmres':
#                 self.iter = sparse.linalg.gmres
#             case 'cg':
#                 self.iter = sparse.linalg.cg
    
#     def __call__(self, xs, bs, maxiter):
#         '''
#         x: torch.Tensor with shape B x 1 x N x N
#         b: torch.Tensor with shape B x N**2
#         '''
#         original_shape = xs.shape
#         with torch.no_grad():
#             x = torch.flatten(xs, 1, -1).cpu().numpy()
#             b = bs.cpu().numpy()
#             Y = []
#             for i in range(original_shape[0]):
#                 y, _ = self.iter(self.A, b[i].squeeze(), x[i].squeeze(), maxiter=maxiter, tol=1e-15)
#                 Y.append(y)
#             Y = torch.from_numpy(np.stack(Y, axis=0).reshape(original_shape))
#             return Y.type_as(xs)           
        
# class EinJacIter(torch.nn.Module):
#     def __init__(self, A, maxiter=5):
#         super().__init__()
        
#         DigA = torch.diagonal(A, dim1=-2, dim2=-1)
#         D = torch.stack([torch.diag(i) for i in DigA], dim=0)
#         M = A - D
#         self.invD = 1.0 / DigA
#         self.M = M.clone().detach().float()
#         self.maxiter = maxiter
#         self.BatchSize, self.Nx = DigA.shape

#     def _step(self, x, B):
#         '''
#             Matrix A: (BatchSize, Nx, Nx)
#             RHS B: (BatchSize, Nx)
#             Input x: (BatchSize, 1, Nx, Nx)
#         '''
#         with torch.no_grad():
#             y = self.invD * (B - torch.einsum('kij,kj->ki', self.M, x))
#             return y
        
#     def fresh_maxiter(self, maxiter):
#         self.maxiter = maxiter

#     def forward(self, x, B):
#         with torch.no_grad():
#             y = x.reshape(self.BatchSize, -1)
#             y = self._step(y, B)
#             for _ in range(self.maxiter):
#                 y = self._step(y, B)
#             return y.reshape(x.shape)

# class OneNLinearSubItr(torch.nn.Module):
#     def __init__(self, method, maxiter, A, b, ans, batch_size, device='cuda'):
#         super().__init__()
#         self.device = device
#         self.maxiter = maxiter
#         self.method = method
#         self._generator(A)
#         b = torch.from_numpy(b).float()
#         self.b = b.expand(batch_size, -1).to(device)
#         self.A = coo2tensor(A.tocoo(), device)
#         self.ans = torch.from_numpy(ans).to(self.device)

#     def _generator(self, A):
#         match self.method:
#             case 'jac':
#                 self.generator = Jac(A, self.device)
#             case 'gmres' | 'cg':
#                 self.generator = NpSolver(A, self.method)
    
#     def fresh_maxiter(self, maxiter):
#         self.maxiter = maxiter
    
#     def forward(self, u):
#         with torch.no_grad():
#             v = self.generator(u, self.b, self.maxiter)
#             return v
        

# class PinnGenerator(torch.nn.Module):
#     def __init__(self, GridSize, device, maxiter, area, gd=0):
#         super().__init__()
#         self.GridSize = GridSize
#         (left, bottom), (right, top) = area
#         self.h = (right - left) / (GridSize - 1)
#         self.device = device
#         self.maxiter = maxiter
#         self.gd = gd

#         self.k1 = self._get_kernel([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
#         self.k2 = self._get_kernel([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
#         self.k3 = self._get_kernel([[0, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 0]])

#     def _get_kernel(self, k):
#         k = torch.tensor(k, requires_grad=True)
#         k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).float().to(self.device)
#         return k

#     def hard_encode(self, x):
#         y = F.pad(x, (1, 1, 1, 1), 'constant', value=self.gd)
#         return y

#     def jac_step(self, x, w, f):
#         u = self.hard_encode(x)
#         force = f[..., 1:-1, 1:-1] * self.h**2
#         y1 = F.conv2d(u, self.k1) * w[..., 1: -1, 1: -1]        
#         y2 = F.conv2d(w * u, self.k2)
#         y3 = F.conv2d(w, self.k3)
#         return (force + y1 + y2) / y3
    
#     def forward(self, x, w, f):
#         with torch.no_grad():
#             y = self.jac_step(x, w, f)
#             for _ in range(self.maxiter):
#                 y = self.jac_step(y, w, f)
#         return y