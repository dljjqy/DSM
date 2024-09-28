from .baseProblem import BaseProblem
from scipy.stats import multivariate_normal
import numpy as np


def normal(x, y, h, mean=[0, 0]):
    var = np.diag([0.5] * 2) * h**2
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, var)
    return rv.pdf(pos)

class WaterPump:
    def __init__(self, locs, Qs, h):
        self.locs = locs
        self.Qs = Qs
        self.h = h
        
    def __call__(self, x, y):
        result = [-Q * normal(x, y, self.h, loc) for Q, loc in zip(self.Qs, self.locs) ]
        return np.stack(result, axis=0).sum(axis=0)

class LinearWaterFlow(BaseProblem):
    def __init__(self, layout, bc_case=1, area=((-250, -250), (250, 250)), eps=1e-9):
        super().__init__()
        self.pump = layout
        self.bc_case = bc_case
        self.left, self.bottom = area[0]
        self.right, self.top = area[1] 
        self.eps = eps

    def diffusive(self, X):
        return np.eye(2)
    
    def sol(self, X):
        return None
    
    def grad(self, X):
        return None
    
    def source(self, X):
        x, y = X[0], X[1]
        return self.pump(x, y)
    
    def is_dirichlet(self, X):
        x, y = X[0], X[1]
        match self.bc_case:
            case 1:
                return x == self.left or x == self.right or y == self.bottom or y == self.top
            case 2:
                return y == self.bottom or y == self.top
            
    def is_neumann(self, X):
        x, y = X[0], X[1]
        match self.bc_case:
            case 1:
                return False
            case 2:
                return x == self.left or x == self.right
                
    def dirichlet(self, X):
        return 100
    
    def neumann(self, X):
        return 0

class HeterWaterFlow(BaseProblem):
    def __init__(self, layout, bc_case=1, area=((-250, -250), (250, 250)), eps=1e-9):
        super().__init__()
        self.pump = layout
        self.bc_case = bc_case
        self.left, self.bottom = area[0]
        self.right, self.top = area[1] 
        self.eps = eps

    def diffusive(self, X):
        x, y = X[0], X[1]
        if x <= 0:
            return 3 * np.array([[10, 5], [5, 30]])
        else:
            return 3 * np.array([[60, 10], [10, 20]])
    
    def sol(self, X):
        return None
    
    def grad(self, X):
        return None
    
    def source(self, X):
        x, y = X[0], X[1]
        return 100 * self.pump(x, y)
    
    def is_dirichlet(self, X):
        x, y = X[0], X[1]
        match self.bc_case:
            case 1:
                return x == self.left or x == self.right or y == self.bottom or y == self.top
            case 2:
                return y == self.bottom or y == self.top
            
    def is_neumann(self, X):
        x, y = X[0], X[1]
        match self.bc_case:
            case 1:
                return False
            case 2:
                return x == self.left or x == self.right
                
    def dirichlet(self, X):
        return 100
    
    def neumann(self, X):
        return 0