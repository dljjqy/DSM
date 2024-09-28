import numpy as np
from .BaseProblem import BaseProblem

def kappa(u, mu):
    return ((3 * u * (1 - u)) / (3 * u**3 + (1-u)**3) )**2 + mu

def force(x, y, center, delta=0.1):
    px, py = center
    mask = (x > px-delta) * (x <= px+delta) * (y > py-delta) * (y <= py+delta)
    force = 100 * np.exp(-50 * ((x - px)**2 + (y - py)**2))
    return mask * force

class NonLinearProblem(BaseProblem):
    def __init__(self, cof, center, *args):
        super().__init__(*args)
        if not cof is None:
            self.cof = cof[..., np.newaxis, np.newaxis] * np.eye(2)[np.newaxis, np.newaxis, ...]
        else:
            self.cof = np.ones((self.Nx, self.Ny, 1, 1)) * np.eye(2)[np.newaxis, np.newaxis, ...]

        xx, yy = np.meshgrid(
            np.arange(self.left + self.dx / 2, self.right, self.dx),
            np.arange(self.bottom + self.dy / 2, self.top, self.dy),
        )
        self.force = force(xx, yy, center)


    def is_dirichlet(self, i, j, edge):
        return self.is_boundary(i, j, edge)

    def is_neumann(self, i, j, edge):
        return False
    
    def dirichlet(self, *args):
        return 0

    def neumann(self, *args):
        return 0