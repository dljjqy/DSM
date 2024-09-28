# Abstract Base Class of numerical schemes for diffuion problems
import numpy as np
import scipy.sparse as sparse
from .utils import *
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import gmres, bicgstab

class UniformFVM():
    def __init__(self, area, Nx, Ny, problem,):
        self.problem = problem
        (left, bottom), (right, top) = area
        self.dx = (right - left) / Nx
        self.dy = (top - bottom) / Ny
        self.Nx = Nx
        self.Ny = Ny
        self.n_faces = Nx * Ny
        self.mu_K = self.mu_L = 0.5
        self.R = np.array([[0, 1], [-1, 0]])
        self.A = lil_matrix((self.n_faces, self.n_faces), dtype=np.float64)
        self.b = np.zeros(self.n_faces)
        self.ua = np.zeros(self.n_faces)

        self.normals = (
             np.array([-1, 0])            
            ,np.array([0, -1])
            ,np.array([1, 0])
            ,np.array([0, 1])
            )
        
        self.dis_vecs = [
             np.array([-self.dx, 0])
            ,np.array([0, -self.dy])
            ,np.array([self.dx, 0])
            ,np.array([0, self.dy])
        ]

        self.uppers = [
            0.5 * self.dx,
            0.5 * self.dy,
            0.5 * self.dx,
            0.5 * self.dy,
        ]

        self.lengths = [
            self.dy,
            self.dx, 
            self.dy,
            self.dx
        ]
    
    def is_boundary(self, i, j, edge):
        match edge:
            case 0:
                return j == 0
            case 1:
                return i == 0
            case 2:
                return j == self.Nx-1
            case 3:
                return i == self.Ny-1
    
    def is_boundary_cell(self, i, j):
        return i == 0 or i == self.Ny-1 or j == 0 or j == self.Nx - 1
            
    def idx(self, i, j):
        return i * self.Nx + j

    def idxL(self, i, j, edge):
        match edge:
            case 0:
                return self.idx(i, j-1)
            case 1:
                return self.idx(i-1, j)  
            case 2:
                return self.idx(i, j+1)
            case 3:
                return self.idx(i+1, j)

    def _construction_vector(self, K, i, j, edge):
        if self.is_boundary(i, j, edge):
            return self.dis_vecs[edge] / 2

        match edge:
            # Left side
            case 0:
                cof = self.problem.cof[i, j-1]
            # Bottom side
            case 1:
                cof = self.problem.cof[i-1, j]
            # Right Side
            case 2:
                cof = self.problem.cof[i, j+1]
            # Top Side
            case 3:
                cof = self.problem.cof[i+1, j]

        n = self.normals[edge]
        lbd = np.dot(n, cof.T @ n)
        dis = self.dis_vecs[edge]
        up = self.uppers[edge]
        # print(f"Edge:{edge}, Normal{n}, lbd:{lbd}, dis:{dis}, up:{up}")

        vec = dis + (up / lbd) * ((K - cof).T @ n)
        return vec
    
    def _construction_vectors(self, i, j):
        K = self.problem.cof[i, j]
        vecs = []
        for e in range(4):
            vecs.append(
                self._construction_vector(K, i, j, e)
                )
        return K, vecs
    
    def _alpha_beta(self, edge, K, vec, vec_prev, vec_next):
        c = self.lengths[edge] * K.T @ self.normals[edge]
        t = self.R @ vec
        alpha_prev = np.dot(c, t) / np.dot(vec_prev, t)
        alpha_next = np.dot(c, t) / np.dot(vec_next, t)

        beta_prev = np.dot(c, self.R @ vec_prev) / np.dot(vec, self.R @ vec_prev)
        beta_next = np.dot(c, self.R @ vec_next) / np.dot(vec, self.R @ vec_next)
        return alpha_prev, beta_prev, alpha_next, beta_next

    def _element_stiffness_matrix(self, i, j):
        AK = np.zeros((4, 4))
        K, vec_KLs = self._construction_vectors(i, j)

        for e in range(4):
            e_prev, e_next = (e+3)%4, (e+1)%4 
            vec = vec_KLs[e]
            vec_prev = vec_KLs[e_prev]
            vec_next = vec_KLs[e_next]

            alpha_prev, beta_prev, alpha_next, beta_next = self._alpha_beta(e, K, vec, vec_prev, vec_next)
            
            if self.problem.is_neumann(i, j, e):
                continue
            else:
                if self.problem.is_neumann(i, j, e_prev):
                    AK[e, e] = beta_next
                    AK[e, e_next] = alpha_next
                    
                elif self.problem.is_neumann(i, j, e_next):
                    AK[e, e] = beta_prev
                    AK[e, e_prev] = alpha_prev
                    
                else:
                    AK[e, e] = (beta_prev + beta_next) / 2
                    AK[e, e_next] = alpha_next / 2
                    AK[e, e_prev] = alpha_prev / 2
        return AK

    def assemble_stiffness_matrix(self):
        """Assemble matrix for all the volumn"""
        for i in range(self.Ny):
            for j in range(self.Nx):
                k = self.idx(i, j)
                AK = self._element_stiffness_matrix(i, j)
                
                for ei in range(4):
                    if not self.is_boundary(i, j, ei):
                        li = self.idxL(i, j, ei)

                        for ej in range(4):
                            self.A[k, k] += self.mu_K * AK[ei, ej]
                            self.A[li,k] -= self.mu_L * AK[ei, ej]
                        
                            if not self.is_boundary(i, j, ej):
                                lj = self.idxL(i, j, ej)
                                self.A[k, lj] -= self.mu_K * AK[ei, ej]
                                self.A[li,lj] += self.mu_L * AK[ei, ej]

                    elif self.problem.is_dirichlet(i, j, ei):
                        for ej in range(4):
                            self.A[k, k] += AK[ei, ej]

                            if not self.is_boundary(i, j, ej):
                                lj = self.idxL(i, j, ej)
                                self.A[k,lj] -= AK[ei, ej]

    def treat_boundary_condition(self):
        for i in range(self.Ny):
            for j in range(self.Nx):
                if self.is_boundary_cell(i, j):
                    k = self.idx(i, j)
                    AK = self._element_stiffness_matrix(i, j)
                    
                    for ei in range(4):    
                        if self.is_boundary(i, j, ei):
                        
                            if self.problem.is_neumann(i, j, ei):
                                gN = self.problem.neumann(i, j, ei)
                                self.b[k] -= (self.lengths[ei] * gN)

                            elif self.problem.is_dirichlet(i, j, ei):
                                for ej in range(4):
                                    if self.problem.is_dirichlet(i, j, ej):
                                        gD = self.problem.dirichlet(i, j, ej)
                                        self.b[k] += (AK[ei, ej] * gD)
                        else:
                            li = self.idxL(i, j, ei)
                            for ej in range(4):
                                if self.problem.is_dirichlet(i, j, ej):
                                    gD = self.problem.dirichlet(i, j, ej)
                                    self.b[k]  += (self.mu_K * AK[ei, ej] * gD)
                                    self.b[li] -= (self.mu_L * AK[ei, ej] * gD)

    def assemble_source_term(self):
        for i in range(self.Ny):
            for j in range(self.Nx):
                k = self.idx(i, j)
                self.b[k] += (self.dx * self.dy * self.problem.force[i, j])
    
    def solve(self, solver_name='gmres', tol=1e-13, maxiter=5000):
        
        self.assemble_stiffness_matrix()
        self.treat_boundary_condition()
        self.assemble_source_term()

        # Transfer full_matrix to csr_matrix when solving the linear system
        A = csr_matrix(self.A)
        b = self.b
        
        match solver_name:
            case 'gmres':            
                self.ua, info = gmres(A, b, tol=tol, maxiter=maxiter)
            
            case 'bicgstab':
                self.ua, info = bicgstab(A, b, tol=tol, maxiter=maxiter)
        
            case _:
                self.ua = sla.spsolve(A, b)
                info = 0
        
        if info > 0:
            print(f'Convergence to tolerance not achieved! (#it = {info})')
        elif info < 0:
            print('Illegal input or breakdown!')
        
        return self.ua
        
    
    def compute_max_error(self):
        problem, ua = self.problem, self.ua
        err_max = -np.inf

        for i in range(self.Ny):
            for j in range(self.Nx):
                k = self.idx(i, j)
                ue_ = problem.ans[i, j]      
                ua_ = ua[k]
                err = np.abs(ue_ - ua_)
                if err > err_max:
                    err_max = err
            return err_max
    
    def compute_l2_error(self):
        problem, ua = self.problem, self.ua
        err_l2 = 0
        
        for i in range(self.Ny):
            for j in range(self.Nx):
                k = self.idx(i, j)
                ue_ = problem.ans[i, j]      
                ua_ = ua[k]
                err = np.abs(ue_ - ua_)

                err_l2 += err ** 2 * self.dx * self.dy
        return np.sqrt(err_l2)

    def compute_h1_error(self):
        problem, ua = self.problem, self.ua
        err_h1 = 0
        
        for i in range(self.Ny):
            for j in range(self.Nx):
                k = self.idx(i, j)
                ue_ = problem.ans[i, j]      
                ua_ = ua[k]
                ek_ = ua_ - ue_
                for e in range(4):
                    if not self.is_boundary(i, j, e):
                        le = self.idxL(i, j, e)

                        li = int(le // self.Nx)
                        lj = int(le % self.Nx)
                        
                        el_ = problem.ans[li, lj] - ua[le]
                        err_h1 += (ek_ - el_) ** 2
        return np.sqrt(err_h1)
            
    def compute_errors(self):
        hmesh = self.dx
        err_max = self.compute_max_error()
        err_l2 = self.compute_l2_error()
        err_h1 = self.compute_h1_error()
        
        return hmesh, err_max, err_l2, err_h1
    
    def is_symmetry(self):
        A = self.A
        L = []        # store the indices of interior vertices        
        for i in range(A.shape[0]):
            if abs(A[i].sum() - A[i,i]) > 1e-13: 
                L.append(i)
        for i in L:
            for j in L:
                if abs(A[i][j] - A[j][i]) > 1e-13:
                    return False
        return True
