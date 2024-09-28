import numpy as np
from .baseProblem import BaseProblem
import torch

def kappa(u, mu):
    return ((3 * u * (1 - u)) / (3 * u**3 + (1-u)**3) )**2 + mu

def kappa_water(u, mu):
    return u + mu

def force(x, y, center, delta=0.05):
    px, py = center
    mask = (x > px-delta) * (x <= px+delta) * (y > py-delta) * (y <= py+delta)
    force = 100 * np.exp(-50 * ((x - px)**2 + (y - py)**2))
    return mask * force

class NLinearProblem(BaseProblem):
    def __init__(self, h, u0, point_center=(0.5, 0.5), area=((0, 0), (1, 1)), mu=0.2, eps = 1e-9):
        super().__init__()
        self.h = h
        self.u0 = u0
        self.area = area
        self.eps = eps
        self.mu = mu
        self.p = point_center
        (self.left, self.bottom), (self.right, self.top) = area
        self._gen_diffusive()
        # xx, yy = np.meshgrid(
        #     np.arange(h/2, self.right, h,),
        #     np.arange(h/2, self.top, h,),
        # )
        if self.u0 is None:
            GridSize = (self.right - self.left) // self.h
            self.cof = np.ones((GridSize, GridSize))
        else:
            self.cof = kappa(u0, self.mu)

    
    def _gen_diffusive(self):
        # self.kappa = kappa
        if self.u0 is None:
            self.diffusive = lambda X: np.eye(2)
        else:
            self.k = kappa(self.u0, self.mu)
            def _diffusive(X):
                x, y = X[0], X[1]
                i, j = int((y - self.bottom) // self.h), int((x - self.left) // self.h)
                return self.k[i, j] * np.eye(2)
            self.diffusive = _diffusive

    def sol(self, X):
        return None
    
    def grad(self, X):
        return None
    
    def source(self, X):
        x, y = X[0], X[1]
        return force(x, y, self.p, 0.05)
        
    def is_dirichlet(self, X):
        x, y = X[0], X[1]
        return x == self.left or x == self.right or y == self.bottom or y == self.top
            
    def is_neumann(self, X):
        return False
                
    def dirichlet(self, X):
        return 0
    
    def neumann(self, X):
        return 0