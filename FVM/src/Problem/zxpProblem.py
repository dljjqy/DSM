#!/usr/bin/env python
import numpy as np
from ..utils import *


class zxpProblem():

    def __init__(self, case=0, eps=None):
        super().__init__()
        self.case = case
        self.eps = eps
        # self.mesh = mesh

    def diffusive(self, X):
        """ Compute the diffusion tensor
        
        Params
        ======
        X (2, ) ndarray

        Return
        ======
        (2, 2) ndarray
        """
        x, y = X[0], X[1]
        match self.case:
            case 0 | 1 | 2 | 3 | 4 | 5:
                return np.eye(2)
        
            case 6:
                if x <= 0.5:
                    return np.array([[3., 1.],
                                    [1., 3.]])
                else:
                    return np.array([[10., 3.],
                                    [3., 10.]])
            
            case 7:
                return np.array([[1.5, 0.5],
                                 [0.5, 1.5]])
        
            case 8:
                if x <= 0.5:
                    return np.eye(2)
                else:
                    return 4*np.eye(2)

            case 9:
                if y < 0.5:
                    if x < 0.5:
                        return np.array([[10, 0],
                                        [0, 0.01]])
                    else:                
                        return np.array([[0.1, 0],
                                        [0, 100.]])
                else:
                    if x < 0.5:
                        return np.array([[100, 0],
                                        [0, 0.1]])
                    else:
                        return np.array([[0.01, 0],
                                        [0, 10]])
                
            case 10 | 11:
                beta = self.eps
                return np.array([[beta*x**2+y**2, (beta-1)*x*y],
                                [(beta-1)*x*y, x**2+beta*y**2]]) / (x**2+y**2)

    def sol(self, X):
        x, y = X[0], X[1]
        
        match self.case:
            case 0:
                return 1.0
            
            case 1:
                return x + y
        
            case 2:
                return x**2 + y**2
            
            case 3:
                return x**3 + y**3

            case 4:
                return x**4 + y**4

            case 5:
                return np.exp(-x**2-y**2)
            
            case 6:
                if x <= 0.5:
                    return 14*x + y
                else:
                    return 4*x + y + 5
                
            case 7:
                xx = 1.-x
                yy = 1.-y
                return 0.5 * (np.sin(xx*yy)/np.sin(1.) + xx**3*yy**2)
            
            case 8:            
                if x <= 0.5:
                    return 1 - 2*y**2 + 4*x*y + 6*x + 2*y
                else:
                    return 3.25 - 2*y**2 + 3.5*y + x*y + 1.5*x
                
            case 9:
                if y < 0.5:
                    if x < 0.5:
                        c = 0.1
                    else:
                        c = 10.
                else:
                    if x < 0.5:
                        c = 0.01
                    else:
                        c = 100
                                    
                return c * np.sin(2*np.pi*x) * np.sin(2*np.pi*y) + 12.
            
            case 10:
                return np.sin(np.pi*x) * np.sin(np.pi*y)

            case 11:
                # just boundary don't have an analytical solution
                return 0

    def grad(self, X):
        x, y = X[0], X[1]

        match self.case:
            case 0:
                return np.array([0, 0])

            case 1:
                return np.array([1, 1])

            case 2:
                return np.array([2*x, 2*y])

            case 3:
                u_x = 3*x**2
                u_y = 3*y**2
                return np.array([u_x, u_y])

            case 4:
                u_x = 4*x**3
                u_y = 4*y**3
                return np.array([u_x, u_y])

            case 5:
                u_x = -2*x*np.exp(-x**2-y**2)
                u_y = -2*y*np.exp(-x**2-y**2)
                return np.array([u_x, u_y])

            case 6:
                if x <= 0.5:
                    u_x = 14
                    u_y = 1
                else:
                    u_x = 4
                    u_y = 1
                return np.array([u_x, u_y])

            case 7:
                x = 1. - x
                y = 1. - y

                u_x = 0.5*(-y * np.cos(x * y) / np.sin(1.) - 3 * x**2 * y** 2)
                u_y = 0.5*(-x * np.cos(x * y) / np.sin(1.) - 2 * x**3 * y)
                return np.array([u_x, u_y])

            case 8:
                if x <= 0.5:
                    u_x = 4*y + 6.
                    u_y = -4*y + 4*x + 2.
                else:
                    u_x = y + 1.5
                    u_y = x - 4*y + 3.5
                return np.array([u_x, u_y])

            case 9:
                if y < 0.5:
                    if x < 0.5:
                        c = 0.1
                    else:
                        c = 10.
                else:
                    if x < 0.5:
                        c = 0.01
                    else:
                        c = 100

                u_x = c * 2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
                u_y = c * 2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
                return np.array([u_x, u_y])

            case 10:
                """Heterogenous rotating anisotropy"""
                beta = self.eps

                u_x = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
                u_y = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
                return np.array([u_x, u_y])

    def source(self, X):
        x, y = X[0], X[1]

        match self.case:
            case 0 | 1:
                return 0.0
        
            case 2:
                return -4.0
            
            case 3:
                return -6.0 * x - 6.0 * y
            
            case 4:
                return -12.0 * x**2 - 12.0 * y**2
            
            case 5:
                return -4 * (x**2 + y**2 - 1) * self.sol(X)
            
            case 6:
                return 0.0
            
            case 7:
                K = self.diffusive(X)
                kxx = K[0, 0]
                kxy = K[0, 1]
                kyy = K[1, 1]

                x = 1.-x
                y = 1.-y
                
                ux = -y*np.cos(x*y)/np.sin(1.) - 3*x**2*y**2
                uy = -x*np.cos(x*y)/np.sin(1.) - 2*x**3*y
                uxx = -y**2*np.sin(x*y)/np.sin(1.) + 6*x*y**2
                uxy = -x*y*np.sin(x*y)/np.sin(1.) + np.cos(x*y)/np.sin(1.) + 6*x**2*y
                uyy = -x**2*np.sin(x*y)/np.sin(1.) + 2*x**3

                return -0.5*(kxx*uxx + 2*kxy*uxy + kyy*uyy)

            case 8:
                K = self.diffusive(X)
                kxx = K[0, 0]
                kxy = K[0, 1]
                kyy = K[1, 1]


                if x <= 0.5:
                    ux = 4*y + 6.
                    uy = -4*y + 4*x + 2.
                    uxx = 0.
                    uxy = 4.
                    uyy = -4.

                else:
                    ux = y + 1.5
                    uy = x - 4*y + 3.5 
                    uxx = 0.
                    uxy = 1.
                    uyy = -4.               

                return -(kxx*uxx + 2*kxy*uxy + kyy*uyy)

            case 9:            
                K = self.diffusive(X)
                kxx = K[0, 0]
                kxy = K[0, 1]
                kyy = K[1, 1]

                if y < 0.5:
                    if x < 0.5:
                        c = 0.1
                    else:
                        c = 10.
                else:
                    if x < 0.5:
                        c = 0.01
                    else:
                        c = 100

                ux = c * 2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
                uy = c * 2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
                uxx = -c * (2*np.pi)**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
                uxy =  c * (2*np.pi)**2 * np.cos(2*np.pi*x) * np.cos(2*np.pi*y)
                uyy = -c * (2*np.pi)**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)

                return -(kxx*uxx + 2*kxy*uxy + kyy*uyy)

            case 10:
                """Heterogenous rotating anisotropy"""
                beta = self.eps
                
                K = self.diffusive(X)
                kxx = K[0, 0]
                kxy = K[0, 1]
                kyy = K[1, 1]

                scale = (beta-1) / (x**2+y**2)**2
                kxx_x = 2*x*y**2 * scale
                kxy_x = y*(y**2-x**2) * scale
                kyx_y = x*(x**2-y**2) * scale
                kyy_y = 2*x**2*y * scale

                ux = np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
                uy = np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)

                uxx = -np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
                uxy =  np.pi**2 * np.cos(np.pi*x) * np.cos(np.pi*y)
                uyy = -np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)

                
                return - ((kxx_x+kyx_y)*ux + (kxy_x+kyy_y)*uy + (kxx*uxx + 2*kxy*uxy + kyy*uyy))

            case 11:
                return 10 if 3/8 <= x <= 5/8 and 3/8 <= y <= 5/8 else 0

    
    # def is_dirichlet(self, X):
    #     x, y = X[0], X[1]

    #     return x == 0 or x == 1 or y == 0 or y == 1

    # def is_neumann(self, X):
    #     return False
    
    def is_dirichlet(self, X):
        x, y = X[0], X[1]
        return x == 1

    def is_neumann(self, X):
        x, y = X[0], X[1]
        return x == 0 or y == 0 or y == 1

    def dirichlet(self, X):
        if self.case == 11:
            return 0
        else:
            return self.sol(X)
    
    def neumann(self, X):
        x, y = X[0], X[1]
        if x == 0:
            n = np.array([-1, 0])
        elif x == 1:
            n = np.array([1, 0])
        elif y == 0:
            n = np.array([0, -1])
        elif y == 1:
            n = np.array([0, 1])

        return -np.dot(self.diffusive(X) @ self.grad(X), n)

    def __repr__(self):
        return f'Poisson problem with {self.case} case'


if __name__ == '__main__':
    problem = zxpProblem(11, eps=1e-1)
    # print(problem.case)

    def limiter(X, alpha=3):
        x, y = X[0], X[1]
        res =  (1 - min(x, y) / max(x, y))**alpha  if x > 0 and y > 0 else 1
        return res * (x - y)

    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    # u = np.array([problem.sol(X[i]) for i in range(len(X))]).reshape(xx.shape)
    N = 7
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    xx, yy = np.meshgrid(x, y)
    X = np.vstack((xx.flatten(), yy.flatten())).T
    f = np.array([limiter(X[i], alpha=N) for i in range(len(X))]).reshape(xx.shape)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    

    # ax.plot_surface(xx, yy, f, rstride=1, cstride=1,
    #             cmap='rainbow', edgecolor='none')
    cs = ax.contour(xx, yy, f, levels=[-2, -1.75, -1.5, -1.25, -1,
                                       -0.75, -0.5, -0.375, -0.25, -0.125,
                                       0, 0.125,  0.25, 0.375,  0.5, 0.75,
                                       1, 1.25, 1.5, 1.75, 2])
    ax.clabel(cs, fontsize=9, inline=True)
    ax.set_aspect('equal') 
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(f'N={N}')

    ax = fig.add_subplot(projection='3d')
    ax.view_init(32, -48)
    plt.show()
