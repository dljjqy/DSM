from .BaseProblem import BaseProblem
import numpy as np

class BlockSourceProblem(BaseProblem):
    def __init__(self, layout, bd_case, gD=298, *args):
        super().__init__(*args)
        self.bd_case = bd_case
        self.gD = gD

        xx, yy = np.meshgrid(
            np.arange(self.left + self.dx / 2, self.right, self.dx),
            np.arange(self.bottom + self.dy / 2, self.top, self.dy),
        )
        self.cof = np.tile(np.eye(2), (self.Nx, self.Ny, 1, 1))
        if not layout is None:
            self.force = layout(xx, yy)

    def is_dirichlet(self, i, j, edge):
        match self.bd_case:
            case 0:
                return self.is_boundary(i, j, edge)
            case 1:
                return self.on_left(i, j, edge)
            case 2:
                x = j * self.dx + self.left + self.dx/2
                is_or_not = self.on_bottom(i, j, edge) and (np.abs(x - 0.05) < 0.001 + 1e-9)
                return is_or_not

    def is_neumann(self, i, j, edge):
        match self.bd_case:
            case 0:
                return False
            case 1:
                return self.is_boundary(i, j, edge) and not self.is_dirichlet(i, j, edge)
            case 2:
                return self.is_boundary(i, j, edge) and not self.is_dirichlet(i, j, edge)

    def dirichlet(self, *args):
        return self.gD

    def neumann(self, *args):
        return 0