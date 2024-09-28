class BaseProblem:
    def __init__(self):
        super().__init__()

    def diffusive(self):
        raise NotImplementedError
    
    def sol(self):
        raise NotImplementedError
    
    def source(self):
        raise NotImplementedError
    
    def is_dirichlet(self):
        raise NotImplementedError

    def is_neumann(self):
        raise NotImplementedError

    def dirichlet(self):
        raise NotImplementedError

    def neumann(self):
        raise NotImplementedError
