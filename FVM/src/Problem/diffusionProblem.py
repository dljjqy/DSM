from .baseProblem import BaseProblem
import numpy as np

class PieceWiseConst:
    def __init__(self, mu, area=((0, 0), (1, 1))):
        self.mu = mu
        self.left, self.bottom = area[0]
        right, top = area[1]
        N, M = self.mu.shape
        self.dx = (right - self.left) / M
        self.dy = (top - self.bottom) / N
    
    def __call__(self, x, y):
        index_x = int((x - self.left) // self.dx)
        index_y = int((y - self.bottom) // self.dy)
        return self.mu[index_y, index_x]

class VaryDiffusionCof(BaseProblem):
    
    def __init__(self, mu, area=((0, 0), (1, 1)), eps=1e-9):
        super().__init__()
        self.mu = mu
        self.area = area
        self.eps = eps
        self._gen_diffusive()

    def _gen_diffusive(self):
        diffusive = PieceWiseConst(self.mu, self.area)
        self.diffusive = lambda X: diffusive(X[0], X[1]) * np.eye(2)

    def fresh_mu(self, mu):
        self.mu = mu
    
    def sol(self, X):
        return None
    
    def grad(self, X):
        return None
    
    def source(self, X):
        x, y = X[0], X[1]
        return 0 * x
    
    def is_dirichlet(self, X):
        x, y = X[0], X[1]
        if np.abs(y - 1.0) <= self.eps:
            return True
        else:
            return False
            
    def is_neumann(self, X):
        x, y = X[0], X[1]
        if np.abs(x) <= self.eps or np.abs(x - 1.0) <= self.eps or np.abs(y) <= self.eps:
            return True 
        else:
            return False
         
    def dirichlet(self, X):
        return 0.0
    
    def neumann(self, X):
        x, y = X[0], X[1]
        if np.abs(x) <= self.eps or np.abs(x - 1.0) <= self.eps:
            return 0.0
        
        else:
            return -1.0
