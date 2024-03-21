import torch
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import random
import pandas as pd
from multiprocessing import Pool
from time import time
from FVM.src.utils import *
from FVM.src.ICD import VolumnCenteredScheme
from FVM.src.Problem import *

import numpy as np
from scipy.stats import multivariate_normal

def worker(task_id, area, GridSize, boundary_gap, chip_gap):
    try:
        info = SeqLS(task_id, area, GridSize, boundary_gap, chip_gap)
        return info  
    except Exception as e:
        # print(e)
        return None  

def layout2csv(DataN, area, GridSize, boundary_gap, chip_gap):
        infos = []
        start = 0
        with Pool() as pool:
            while len(infos) < DataN:
                remaining = DataN - len(infos)
                print(f'{len(infos)/DataN * 100:.2f}%', flush=True)
                tasks = range(start, start + remaining)
                results = pool.starmap(worker, [(task, area, GridSize, boundary_gap, chip_gap) for task in tasks])
                infos.extend(filter(None, results))
                start = remaining + start 

        print(len(infos))
        infos = np.vstack(infos)
        dic = {
            'idx': 'int',
            'x':'float', 
            'y':'float', 
            'w':'float', 
            'h':'float', 
            'c':'float'
        }
        df = pd.DataFrame(infos, columns=dic.keys()).astype(dic)
        return df

def SeqLS(idx, area=((0, 0), (0.1, 0.1)), GridSize=500, boundary_gap=0.001, chip_gap=0.001):
    random.seed(idx+ 100 + int(time() * 1000))
    (left, bottom), (right, top) = area
    vx = np.linspace(left , right , GridSize)
    vy = np.linspace(bottom , top, GridSize)
    xx, yy = np.meshgrid(vx, vy)
    info = []

    chips = [
    [0.016, 0.012, 4000], [0.012, 0.006, 16000], [0.018, 0.009, 6000], [0.018, 0.012, 8000],
    [0.018, 0.018, 10000], [0.012, 0.012, 14000],[0.018, 0.006, 16000], [0.009, 0.009, 20000],
    [0.006, 0.024, 8000], [0.006, 0.012, 16000], [0.012, 0.024, 10000], [0.024, 0.024, 20000]]

    for i, chip in enumerate(chips):
        w, h, c = chip
        def _func0(x, y):
            sign_x = (x >= left + boundary_gap) & (x < right - boundary_gap - w)
            sign_y = (y >= bottom + boundary_gap) & (y < top - boundary_gap - h)
            sign = sign_x & sign_y
            return ~sign
        eVEMs = []
        eVEMs.append(_func0(xx, yy))

        for j in range(i):
            _, xj, yj, wj, hj, _ = info[j]
            l = xj - w - chip_gap
            r = xj + wj + chip_gap
            b = yj - h - chip_gap
            t = yj + hj + chip_gap

            def _funcj(x, y):
                sign_x = (x >= l) & (x < r)
                sign_y = (y >= b) & (y < t)
                sign = sign_x & sign_y
                return sign
            eVEMs.append(_funcj(xx, yy))
        IeVEM = np.sum(np.stack(eVEMs, axis=0), axis=0)
        legal_indices = np.argwhere(IeVEM == 0)

        if len(legal_indices) > 0:
            idx_x, idx_y = random.choice(legal_indices)
            info.append((idx, vx[idx_y], vy[idx_x], w, h, c))
        else:
            raise ValueError
    return info

def coo2tensor(A, device, dtype=torch.float32):
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    
    v = torch.DoubleTensor(values)
    a =  torch.sparse_coo_tensor(
        i, v, A.shape, 
        device=device, requires_grad=False, dtype=dtype, is_coalesced=True)
    return a

def mmbv(A, y):
    """
    Sparse matrix multiply Batched vectors
    """
    y = torch.transpose(y, 0, 1)
    v = torch.sparse.mm(A, y)
    return v.transpose(0, 1)

def bvi(x, y):
    """
    inner product of Batched vectors x and y
    """
    b, n = x.shape[:2]
    inner_values =  torch.bmm(x.view((b, 1, n)), y.view((b, n, 1))) 
    return inner_values.reshape(b, 1)


def reaction_A(n, a):
    '''For Dirichlet boundary condition
    '''
    n2 = n**2
    A = sparse.diags([1] * n2).tolil()
    for i in range(1, n-1):
        for j in range(1, n-1):
            idx = i * n + j
            A[idx, idx]   = 4 * a[i, j] + a[i, j+1] + a[i, j-1] + a[i+1, j] + a[i-1, j]
            A[idx, idx+1] = -(a[i, j] + a[i, j+1]) 
            A[idx, idx-1] = -(a[i, j] + a[i, j-1]) 
            A[idx, idx-n] = -(a[i, j] + a[i-1, j]) 
            A[idx, idx+n] = -(a[i, j] + a[i+1, j]) 
    return A

def reaction_b_dir(f, gd, h):
    '''for gd is constant
    '''
    h2 = 2 * h**2
    b = np.ones_like(f) * gd
    b[1:-1, 1:-1] *= 0
    b[1:-1, 1:-1] += (f[1:-1, 1:-1] * h2)
    return b.reshape(-1)

def reaction_A(n, a):
    '''For Dirichlet boundary condition
    '''
    n2 = n**2
    A = sparse.diags([1] * n2).tolil()
    for i in range(1, n-1):
        for j in range(1, n-1):
            idx = i * n + j
            A[idx, idx]   = 4 * a[i, j] + a[i, j+1] + a[i, j-1] + a[i+1, j] + a[i-1, j]
            A[idx, idx+1] = -(a[i, j] + a[i, j+1]) 
            A[idx, idx-1] = -(a[i, j] + a[i, j-1]) 
            A[idx, idx-n] = -(a[i, j] + a[i-1, j]) 
            A[idx, idx+n] = -(a[i, j] + a[i+1, j]) 
    return A

def reaction_b_dir(f, gd, h):
    '''for gd is constant
    '''
    h2 = 2 * h**2
    b = np.ones_like(f) * gd
    b[1:-1, 1:-1] *= 0
    b[1:-1, 1:-1] += (f[1:-1, 1:-1] * h2)
    return b.reshape(-1)

def nlinear_force(x, y, center, delta=0.05):
    px, py = center
    mask = (x > px-delta) * (x <= px+delta) * (y > py-delta) * (y <= py+delta)
    force = 100 * np.exp(-50 * ((x - px)**2 + (y - py)**2))
    return mask * force
    # return force

def kappa(u, mu):
    return ((3 * u * (1 - u)) / (3 * u**3 + (1-u)**3) )**2 + mu

def kappa_water(u, mu):
    return u + mu

def fd_solve_nlinear(GridSize, area, mu, center=(0.5, 0.5), Picard_maxiter=1000):
    (left, bottom), (right, top) = area
    xx, yy = np.meshgrid(
        np.linspace(left, right, GridSize),
        np.linspace(bottom, top, GridSize)
    )
    h = (right - left) / (GridSize - 1)
    
    b = reaction_b_dir(force(xx, yy, center), 0.0, h)
    A0 = reaction_A(GridSize, np.ones((GridSize, GridSize))).tocsr()
    u0 = spsolve(A0, b)

    for i in range(Picard_maxiter):
        newA = reaction_A(GridSize, kappa(u0.reshape(GridSize, GridSize), mu)).tocsr()
        newu = spsolve(newA, b)

        # delta = ((newu - u0)**2 * h**2).sum()
        delta = np.linalg.norm(newu - u0)
        # error = ((newA @ u0 - b)**2 * h**2).sum()
        error = np.linalg.norm(newA @ u0 - b)
        # print(f"Itr: {i}\t Delta: {delta:.3e}\t Error: {error:.3e}\t")

        if delta < 1e-7 or error < 1e-7:
            break
        else:
            u0, A0 = newu, newA
    return newu.reshape(GridSize, GridSize)

def fvm_solve_nlinear(GridSize, area, mu, center=(0.5, 0.5), Picard_maxiter=1000):
    (left, bottom), (right, top) = area
    h = (right - left) / GridSize

    mesh = readmesh(f'./FVM/my_meshes/UniformQuad-VaryK-{GridSize}.obj')
    solver = VolumnCenteredScheme(mesh)

    # u0 = torch.rand((GridSize, GridSize)).float()
    u0 = np.random.rand(GridSize, GridSize)
    problem = NLinearProblem(h, u0, center, area, mu)
    solver.solve(problem, solver_name = None)
    u0,  b = solver.ua, solver.b

    for i in range(Picard_maxiter):
        new_problem = NLinearProblem(h, u0.reshape(GridSize, GridSize), (0.5, 0.5), area, mu)
        newA = solver.get_A(new_problem).tocsr()
        newu = spsolve(newA, b)

        # delta = ((newu - u0)**2 * h**2).sum()
        delta = np.linalg.norm(newu - u0)
        # error = ((newA @ u0 - b)**2 * h**2).sum()
        error = np.linalg.norm(newA @ u0 - b)
        print(f"Itr: {i}\t Delta: {delta:.3e}\t Error: {error:.3e}\t")

        if delta < 1e-9 or error < 1e-9:
            break
        else:
            u0, A0 = newu, newA
    return newu.reshape(GridSize, GridSize)

def normal(x, y, h, mean=[0, 0]):
    var = np.diag([0.5] * 2) * h**2
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, var)
    return rv.pdf(pos)

def R(x, y):
    return np.sqrt(x**2 + y**2)

def yita11_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask * ((6 * (3 - 4 * r)/np.pi)/H2)

def yita12_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask * ((12 * (5 * r**2 - 8 * r + 3)/np.pi)/H2)

def yita22_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((12 * (15 * r**2 - 20 * r + 6)/np.pi)/H2)

def yita23_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((-60 * (7 * r**3 - 15 * r**2 + 10 * r - 2)/np.pi)/H2)

def yita25_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*((84*(24*r**5-70*r**4+70*r**3-25*r**2+1)/np.pi)/H2)

def yita2cos_2d(x, y, H):
    mask = (R(x, y) <= H)
    r = R(x, y) / H
    H2 = H**2
    return mask*(((-1 / (9*np.pi**4-104*np.pi**2+48)) *
                  ((81*np.pi/16)*(3*np.pi**4-32*np.pi**2+48)*np.cos(3*np.pi*r) +
                   2*np.pi*(9*np.pi**4-80*np.pi**2+48) * np.cos(2*np.pi*r) +
                   ((np.pi/16)*(45*np.pi**2+32*np.pi**2-48)*np.cos(np.pi*r) + 144*np.pi)))/H2)

class L2Loss(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
    
    def forward(self, x, y):
        x = torch.squeeze(x)
        y = torch.squeeze(y)

        diff = (x - y) ** 2 * self.h**2
        errors = torch.sum(diff, dim=0)
        return errors.mean()

class PieceWiseConst:
    def __init__(self, mu, area=((0, 0), (1, 1))):
        self.mu = mu
        left, bottom = area[0]
        right, top = area[1]
        N, M = self.mu.shape
        dx = (right - left) / M
        dy = (top - bottom) / N

        self.token_x = np.arange(left, right+dx, dx)
        self.token_y = np.arange(bottom, top+dx, dy)
    
    def __call__(self, x, y):
        conds = []
        values = []
        for i in range(len(self.token_x) - 1):
            for j in range(len(self.token_y) - 1):
                x0, y0 = self.token_x[i], self.token_y[j]
                x1, y1 = self.token_x[i+1], self.token_y[j+1]
                conds.append(
                    (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
                )
                values.append(self.mu[-j-1, i] * np.ones_like(x))
        return np.select(conds, values, default=0)     

class ChipLayout: 
    def __init__(self, info):
        self.info = info
    
    def __call__(self, X, Y):
        heat = np.zeros_like(X)
        for xi, yi, wi, hi, ci in self.info:
            sign_x = (X >= xi) & (X < xi + wi)
            sign_y = (Y >= yi) & (Y < yi + hi)
            sign = sign_x & sign_y
            heat += (ci * sign)
        return heat
