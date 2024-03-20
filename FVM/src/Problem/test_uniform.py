#!/usr/bin/env python
import numpy as np
# from ..utils import *


class zxpProblem():

    def __init__(self, Nx, Ny, area=((0, 0), (1, 1)), case=0, eps=None):
        super().__init__()
        self.case = case
        self.eps = eps
        ((self.left, self.bottom), (self.right, self.top)) = area
        self.Nx = Nx
        self.Ny = Ny
        self.dx = (self.right - self.left) / self.Nx
        self.dy = (self.top - self.bottom) / self.Ny

        cof = np.zeros((Ny, Nx, 2, 2))
        force = np.zeros((Ny, Nx))
        ans = np.zeros((Ny, Nx))
        for i in range(Ny):
            for j in range(Nx):
                x, y = self.get_loc(i, j)
                cof[i, j] += self.diffusive(x, y)
                force[i, j] += self.source(x, y)
                ans[i, j] += self.sol(x, y)
        self.cof = cof
        self.force = force
        self.ans = ans


    def get_loc(self, i, j):
        return (self.left + j * self.dx + self.dx / 2,
                self.bottom + i * self.dy + self.dy / 2) 
    
    def get_edge_loc(self, i, j, edge):
        x, y = self.get_loc(i, j)
        match edge:
            case 0:
                return (x - self.dx/2, y)
            case 1:
                return (x, y - self.dy/2)
            case 2:
                return (x + self.dx/2, y)
            case 3:
                return (x, y + self.dy/2)
            

    def diffusive(self, x, y):
        """ Compute the diffusion tensor
        
        Params
        ======
        X (2, ) ndarray

        Return
        ======
        (2, 2) ndarray
        """
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

    def sol(self, x, y):
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

    def grad(self, i, j):
        x, y = self.get_loc(i, j)

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

    def source(self, x, y):

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
                return -4 * (x**2 + y**2 - 1) * self.sol(x, y)
            
            case 6:
                return 0.0
            
            case 7:
                K = self.diffusive(x, y)
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
                K = self.diffusive(x, y)
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
                K = self.diffusive(x, y)
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
                
                K = self.diffusive(x, y)
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

    def is_dirichlet(self, i, j, edge):
        match edge:
            case 0:
                return j == 0
            case 1:
                return i == 0
            case 2:
                return j == self.Nx-1
            case 3:
                return i == self.Ny-1


    def is_neumann(self, i, j, edge):
        return False
    
    # def is_dirichlet(self, X):
    #     x, y = X[0], X[1]
    #     return x == 1

    # def is_neumann(self, X):
    #     x, y = X[0], X[1]
    #     return x == 0 or y == 0 or y == 1

    def dirichlet(self,  i, j, edge):
        x, y = self.get_edge_loc(i, j, edge)
        if self.case == 11:
            return 0
        else:
            return self.sol(x, y)
    
    def neumann(self,  i, j, edge):
        x, y = self.get_edge_loc(i, j, edge)
        if x == 0:
            n = np.array([-1, 0])
        elif x == 1:
            n = np.array([1, 0])
        elif y == 0:
            n = np.array([0, -1])
        elif y == 1:
            n = np.array([0, 1])

        return -np.dot(self.diffusive(i, j) @ self.grad(i, j), n)

    def __repr__(self):
        return f'Poisson problem with {self.case} case'

