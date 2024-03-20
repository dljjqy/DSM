# Abstract Base Class of numerical schemes for diffuion problems
import numpy as np
import scipy.sparse as sparse
from .utils import *
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import gmres, bicgstab

class VolumnCenteredScheme():
    def __init__(self, mesh, problem=None):
        self.problem = problem
        self.mesh = mesh

        self.n_faces = mesh.n_faces()
        self.mu_K = self.mu_L = 0.5
        self.R = np.array([[0, 1], [-1, 0]])
        self.A = lil_matrix((self.n_faces, self.n_faces), dtype=np.float64)
        self.b = np.zeros(self.n_faces)
        self.ua = np.zeros(self.n_faces)

    def reboot(self, problem=None):
        if problem is not None:
            self.problem = problem
        self.A *= 0
        self.b *= 0
        self.ua*= 0
        
    def _construction_vector(self, heh, K):
        sigma = edge_mipoint(self.mesh, heh)
        vec_KL = sigma - K
        LambdaK = self.problem.diffusive(K)
        vec_n = normal_vector(self.mesh, heh)

        if is_boundary(self.mesh, heh):    
            return vec_KL
        
        L = adjacent_center(self.mesh, heh)
        LambdaL = self.problem.diffusive(L)

        Lambda_L_sigma = np.dot(vec_n, LambdaL.T@vec_n)
        vec_KL = L - K + np.dot((L-sigma), vec_n) * ((LambdaK.T - LambdaL.T) @ vec_n) / Lambda_L_sigma
        return vec_KL
    
    def _alpha_beta(self, heh, vec_KL, vec_KL_prev, vec_KL_next):
        fh_K = self.mesh.face_handle(heh)

        K = self.mesh.calc_face_centroid(fh_K)[:2]
        LambdaK = self.problem.diffusive(K)
        vec_n = normal_vector(self.mesh, heh)
        length = self.mesh.calc_edge_length(heh)

        c = length * LambdaK.T @ vec_n

        up = np.dot(c, self.R @ vec_KL)
        a_prev = up / np.dot(vec_KL_prev, self.R@vec_KL)
        a_next = up / np.dot(vec_KL_next, self.R@vec_KL)

        beta_prev = np.dot(c, self.R @ vec_KL_prev) / np.dot(vec_KL, self.R@vec_KL_prev)
        beta_next = np.dot(c, self.R @ vec_KL_next) / np.dot(vec_KL, self.R@vec_KL_next)
        return a_prev, a_next, beta_prev, beta_next

    def _element_stiffness_matrix(self, fh):
        N = fh_n_edges(self.mesh, fh)
        AK = np.zeros((N, N))

        K = self.mesh.calc_face_centroid(fh)[:2]
        vec_KLs = [self._construction_vector(heh, K) for heh in self.mesh.fh(fh)]

        for i, heh_i in enumerate(self.mesh.fh(fh)):
            a_prev, a_next, beta_prev, beta_next = self._alpha_beta(heh_i, vec_KLs[i], vec_KLs[(i-1+N)%N], vec_KLs[(i+1)%N])
            if self.problem.is_neumann(edge_mipoint(self.mesh, heh_i)):
                continue
            else:
                prev_heh, next_heh = prev_halfedge(self.mesh, heh_i), next_halfedge(self.mesh, heh_i)
                if self.problem.is_neumann(edge_mipoint(self.mesh, prev_heh)):
                    AK[i, i] = beta_next
                    AK[i, (i+1) % N] = a_next
                    
                elif self.problem.is_neumann(edge_mipoint(self.mesh, next_heh)):
                    AK[i, i] = beta_prev
                    AK[i, (i-1 + N) % N] = a_prev
                    
                else:
                    AK[i, i] = (beta_prev + beta_next) / 2
                    AK[i, (i+1) % N] = a_next / 2
                    AK[i, (i-1+N) % N] = a_prev / 2
        return AK

    def assemble_stiffness_matrix(self):
        """Assemble matrix for all the volumn"""
        for fh in self.mesh.faces():
            k = self.mesh.face_property('global_index', fh)
            AK = self._element_stiffness_matrix(fh)
            for i, heh_i in enumerate(self.mesh.fh(fh)):
                if not is_boundary(self.mesh, heh_i):
                    fh_Li = adjacent_face(self.mesh, heh_i)
                    li = self.mesh.face_property('global_index', fh_Li)
                    for j, heh_j in enumerate(self.mesh.fh(fh)):
                        self.A[k, k] += self.mu_K * AK[i, j]
                        self.A[li,k] -= self.mu_L * AK[i, j]
                            
                        if not is_boundary(self.mesh, heh_j):
                            fh_Lj = adjacent_face(self.mesh, heh_j)
                            lj = self.mesh.face_property('global_index', fh_Lj)
                            self.A[k,lj] -= self.mu_K * AK[i, j]
                            self.A[li,lj]+= self.mu_L * AK[i, j]

                elif self.problem.is_dirichlet(edge_mipoint(self.mesh, heh_i)):
                    for j, heh_j in enumerate(self.mesh.fh(fh)):
                        self.A[k, k] += AK[i, j]

                        if not is_boundary(self.mesh, heh_j):
                            fh_Lj = adjacent_face(self.mesh, heh_j)
                            lj = self.mesh.face_property('global_index', fh_Lj)
                            self.A[k,lj] -= AK[i, j]

    def treat_boundary_condition(self):
        for fh in self.mesh.faces():
            if self.mesh.is_boundary(fh):
                k = self.mesh.face_property('global_index', fh)
                AK = self._element_stiffness_matrix(fh)
                for i, heh_i in enumerate(self.mesh.fh(fh)):
                    if is_boundary(self.mesh, heh_i):
                        if self.problem.is_neumann(edge_mipoint(self.mesh, heh_i)):
                            gN = self.problem.neumann(edge_mipoint(self.mesh, heh_i))
                            self.b[k] -= self.mesh.calc_edge_length(heh_i) * gN
                        
                        elif self.problem.is_dirichlet(edge_mipoint(self.mesh, heh_i)):  
                            for j, heh_j in enumerate(self.mesh.fh(fh)):
                                if self.problem.is_dirichlet(edge_mipoint(self.mesh, heh_j)):
                                    gD = self.problem.dirichlet(edge_mipoint(self.mesh, heh_j))
                                    self.b[k] += (AK[i, j] * gD)
                    else:
                        fh_Li = adjacent_face(self.mesh, heh_i)
                        li = self.mesh.face_property('global_index', fh_Li)
                        for j, heh_j in enumerate(self.mesh.fh(fh)):
                            if self.problem.is_dirichlet(edge_mipoint(self.mesh, heh_j)):
                                gD = self.problem.dirichlet(edge_mipoint(self.mesh, heh_j))
                                self.b[k] += (self.mu_K * AK[i, j] * gD)
                                self.b[li]-= (self.mu_L * AK[i, j] * gD)

    def assemble_source_term(self):
        for fh in self.mesh.faces():
            K = self.mesh.calc_face_centroid(fh)[:2]
            fK = self.problem.source(K)
            area_K = area(self.mesh, fh)
            k = self.mesh.face_property('global_index', fh)
            self.b[k] += (fK * area_K) 

    def get_A(self, problem):
        self.A = lil_matrix((self.n_faces, self.n_faces), dtype=np.float64)
        self.problem = problem
        self.assemble_stiffness_matrix()
        return self.A
    
    def get_b(self, problem):
        self.b *= 0
        self.problem = problem
        
        self.treat_boundary_condition()
        # self.assemble_source_term()
        return np.copy(self.b)
    
    def solve(self, problem=None, solver_name='gmres', tol=1e-13, maxiter=5000):
        
        self.reboot(problem)
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
        mesh, problem, ua = self.mesh, self.problem, self.ua
        err_max = -np.inf

        for fh in mesh.faces():
            i = mesh.face_property('global_index', fh)
            k = mesh.calc_face_centroid(fh)[:2]
            ue_ = problem.sol(k)      
            ua_ = ua[i]
            err = np.abs(ue_ - ua_)
            if err > err_max:
                err_max = err
        return err_max
    
    def compute_l2_error(self):
        mesh, problem, ua = self.mesh, self.problem, self.ua
        err_l2 = 0
        
        for fh in mesh.faces():
            i = mesh.face_property('global_index', fh)
            k = mesh.calc_face_centroid(fh)[:2]
            ue_ = problem.sol(k)
            ua_ = ua[i]
            err = np.abs(ue_ - ua_)

            err_l2 += err ** 2 * area(self.mesh, fh)
        return np.sqrt(err_l2)

    def compute_h1_error(self):
        mesh, problem, ua = self.mesh, self.problem, self.ua
        err_h1 = 0
        for fh in mesh.faces():
            i = mesh.face_property('global_index', fh)
            k = mesh.calc_face_centroid(fh)[:2]
            ue_ = problem.sol(k)
            ua_ = ua[i]
            ek_ = ua_ - ue_
            for heh in mesh.fh(fh):
                if not is_boundary(mesh, heh):
                    fh_L = adjacent_face(mesh, heh)
                    L = adjacent_center(mesh, heh)
                    j = mesh.face_property('global_index', fh_L)
                    el_ = problem.sol(L) - ua[j]
                    err_h1 += (ek_ - el_) ** 2
        return np.sqrt(err_h1)
            
    def compute_errors(self):
        mesh = self.mesh
        
        hmesh = np.max([mesh.face_property('diameter', fh) for fh in mesh.faces()])
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
