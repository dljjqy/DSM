
class BaseProblem:
    def __init__(self, N, area):
        (self.left, self.bottom), (self.right, self.top) = area
        self.Nx = self.Ny = N
        self.dx = (self.right - self.left) / self.Nx
        self.dy = (self.top - self.bottom) / self.Ny
        pass
    
    def on_left(self, i, j, e):
        if j == 0 and e == 0:
            return True
        else:
            return False
    
    def on_bottom(self, i, j, e):
        if i == 0 and e == 1:
            return True
        else:
            return False
    
    def on_right(self, i, j, e):
        if j == self.Nx - 1 and e == 2:
            return True
        else:
            return False
    
    def on_top(self, i, j, e):
        if i == self.Ny - 1 and e == 3:
            return True
        else:
            return False
    
    def is_boundary(self, i, j, e):
        match e:
            case 0:
                return self.on_left(i, j, e)
            case 1:
                return self.on_bottom(i, j, e)
            case 2:
                return self.on_right(i, j, e)
            case 3:
                return self.on_top(i, j, e)