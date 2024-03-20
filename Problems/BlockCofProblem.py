from .BaseProblem import BaseProblem
# from ..utils import PieceWiseConst
import numpy as np


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


# def expand(mu, Nx, Ny):
#     Mx, My = mu.shape
#     col_times = int(Ny // My) + 1
#     row_times = int(Nx // Mx) + 1

#     expanded_mu = np.repeat(mu, row_times, axis=1)
#     expanded_mu = np.repeat(expanded_mu, col_times, axis=0)


#     return expanded_mu[:Ny, :Nx]

class BlockCofProblem(BaseProblem):
    def __init__(self, mu, *args):
        super().__init__(*args)
        if mu.shape[0] in [2, 3, 4]:
            pwc = PieceWiseConst(mu, self.area)
            xx, yy = np.meshgrid(
                np.arange(self.left + self.dx/2, self.right, self.dx),
                np.arange(self.bottom + self.dy/2, self.top, self.dy)
            )
            cof = pwc(xx, yy)
        else:
            cof = mu
        self.cof = cof[..., np.newaxis, np.newaxis] *\
            np.eye(2)[np.newaxis, np.newaxis, ...]
        
        self.force = np.zeros((self.Nx, self.Ny))

    def is_dirichlet(self, i, j, edge):
        return self.on_top(i, j, edge)
    
    def is_neumann(self, i, j, edge):
        return self.is_boundary(i, j, edge) and not self.on_top(i, j, edge)
         
    def dirichlet(self, i, j, e):
        return 0.0
    
    def neumann(self, i, j, e):
        if self.on_left(i, j, e) or self.on_right(i, j, e):
            return 0.0
        elif self.on_bottom(i, j, e):
            return -1.0
