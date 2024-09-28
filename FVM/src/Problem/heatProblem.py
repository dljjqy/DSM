from .baseProblem import BaseProblem
import numpy as np

class ChipHeatDissipation(BaseProblem):
    
    def __init__(self, layout, bc_case=1, eps=1e-9):
        super().__init__()
        self.heat = layout
        self.bc_case = bc_case
        self.eps = eps

    def diffusive(self, X):
        return np.eye(2)
    
    def sol(self, X):
        return None
    
    def grad(self, X):
        return None
    
    def source(self, X):
        x, y = X[0], X[1]
        return self.heat(x, y)
    
    def is_dirichlet(self, X):
        x, y = X[0], X[1]
        match self.bc_case:
            case 1:
                return x == 0 or np.abs(x - 0.1) <= self.eps or y == 0 or np.abs(y - 0.1) <= self.eps
            case 2:
                return x == 0     
            case 3:
                return y == 0 and x >= 0.0495 and x <= 0.0505
            
    def is_neumann(self, X):
        x, y = X[0], X[1]
        match self.bc_case:
            case 1:
                return False
            case 2:
                return np.abs(x - 0.1) <= self.eps or np.abs(y - 0.0) <= self.eps or np.abs(y - 0.1) <= self.eps
            case 3:
                return (y == 0 and (x < 0.0495 or x > 0.0505)) or np.abs(y - 0.1) <= self.eps or np.abs(x - 0.1) <= self.eps or x == 0
                    
    def dirichlet(self, X):
        return 298
    
    def neumann(self, X):
        return 0
    
class NormChipHeatDissipation(BaseProblem):
    
    def __init__(self, layout, bc_case=1, eps=1e-9):
        super().__init__()
        self.heat = layout
        self.bc_case = bc_case
        self.eps = eps
    def diffusive(self, X):
        return np.eye(2)
    
    def sol(self, X):
        return None
    
    def grad(self, X):
        return None
    
    def source(self, X):
        x, y = X[0], X[1]
        return self.heat(x, y)
    
    def is_dirichlet(self, X):
        x, y = X[0], X[1]
        match self.bc_case:
            case 1:
                return x == 0 or np.abs(x - 0.1) <= self.eps or y == 0 or np.abs(y - 0.1) <= self.eps
            case 2:
                return x == 0     
            case 3:
                return y == 0 and x >= 0.0495 and x <= 0.0505

            
    def is_neumann(self, X):
        x, y = X[0], X[1]
        match self.bc_case:
            case 1:
                return False
            case 2:
                return np.abs(x - 0.1) <= self.eps or np.abs(y - 0.0) <= self.eps or np.abs(y - 0.1) <= self.eps
            case 3:               \
                return (y == 0 and (x < 0.0495 or x > 0.0505)) or np.abs(y - 0.1) <= self.eps or np.abs(x - 0.1) <= self.eps or x == 0

    def dirichlet(self, X):
        return 0
    
    def neumann(self, X):
        return 0