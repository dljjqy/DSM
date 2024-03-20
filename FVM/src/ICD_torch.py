# Abstract Base Class of numerical schemes for diffuion problems
# import numpy as np
# import scipy.sparse as sparse
from .utils import *
# from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
# import scipy.sparse.linalg as sla
# from scipy.sparse.linalg import gmres, bicgstab
import torch.sparse as sparse
import torch

class VolumnCenteredScheme():
    def __init__(self, mesh, problem=None):
        self.problem = problem
        self.mesh = mesh

        self.n_faces = mesh.n_faces()
        self.mu_K = self.mu_L = 0.5
        self.R = torch.tensor([[0, 1], [-1, 0]]).float().to('cuda')
        self.A = torch.zeros(self.n_faces, self.n_faces).to('cuda')
        self.b = torch.zeros(self.n_faces).to('cuda')
        self.ua = torch.zeros(self.n_faces).to('cuda')

    def reboot(self, problem=None):
        if problem is not None:
            self.problem = problem
        self.A *= 0
        self.b *= 0
        self.ua*= 0
        
    def _construction_vector(self, heh, K):
        sigma = torch.from_numpy(edge_mipoint(self.mesh, heh)).float().to('cuda')
        vec_KL = sigma - K
        LambdaK = self.problem.diffusive(K)
        vec_n = torch.from_numpy(normal_vector(self.mesh, heh)).float().to('cuda')

        if is_boundary(self.mesh, heh):    
            return vec_KL
        
        L = torch.from_numpy(adjacent_center(self.mesh, heh)).float().to('cuda')
        LambdaL = self.problem.diffusive(L)

        Lambda_L_sigma = torch.dot(vec_n, LambdaL.T@vec_n)
        vec_KL = L - K + torch.dot((L-sigma), vec_n) * ((LambdaK.T - LambdaL.T) @ vec_n) / Lambda_L_sigma
        return vec_KL
    
    def _alpha_beta(self, heh, vec_KL, vec_KL_prev, vec_KL_next):
        fh_K = self.mesh.face_handle(heh)

        K = torch.from_numpy(self.mesh.calc_face_centroid(fh_K)[:2]).float().to('cuda')
        LambdaK = self.problem.diffusive(K)
        vec_n = torch.from_numpy(normal_vector(self.mesh, heh)).float().to('cuda')
        length = self.mesh.calc_edge_length(heh)

        c = length * LambdaK.T @ vec_n

        up = torch.dot(c, self.R @ vec_KL)
        a_prev = up / torch.dot(vec_KL_prev, self.R@vec_KL)
        a_next = up / torch.dot(vec_KL_next, self.R@vec_KL)

        beta_prev = torch.dot(c, self.R @ vec_KL_prev) / torch.dot(vec_KL, self.R@vec_KL_prev)
        beta_next = torch.dot(c, self.R @ vec_KL_next) / torch.dot(vec_KL, self.R@vec_KL_next)
        return a_prev, a_next, beta_prev, beta_next

    def _element_stiffness_matrix(self, fh):
        N = fh_n_edges(self.mesh, fh)
        AK = torch.zeros((N, N))

        K = torch.from_numpy(self.mesh.calc_face_centroid(fh)[:2]).float().to('cuda')
        vec_KLs = [self._construction_vector(heh, K) for heh in self.mesh.fh(fh)]

        for i, heh_i in enumerate(self.mesh.fh(fh)):
            a_prev, a_next, beta_prev, beta_next = self._alpha_beta(heh_i, vec_KLs[i], vec_KLs[(i-1+N)%N], vec_KLs[(i+1)%N])
            if self.problem.is_neumann(torch.from_numpy(edge_mipoint(self.mesh, heh_i)).float().to('cuda')):
                continue
            else:
                prev_heh, next_heh = prev_halfedge(self.mesh, heh_i), next_halfedge(self.mesh, heh_i)
                if self.problem.is_neumann(torch.from_numpy(edge_mipoint(self.mesh, prev_heh)).float().to('cuda')):
                    AK[i, i] = beta_next
                    AK[i, (i+1) % N] = a_next
                    
                elif self.problem.is_neumann(torch.from_numpy(edge_mipoint(self.mesh, next_heh)).float().to('cuda')):
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

                elif self.problem.is_dirichlet(torch.from_numpy(edge_mipoint(self.mesh, heh_i)).float().to('cuda')):
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
                        if self.problem.is_neumann(torch.from_numpy(edge_mipoint(self.mesh, heh_i)).float().to('cuda')):
                            gN = self.problem.neumann(torch.from_numpy(edge_mipoint(self.mesh, heh_i)).float().to('cuda'))
                            self.b[k] -= self.mesh.calc_edge_length(heh_i) * gN
                        
                        elif self.problem.is_dirichlet(torch.from_numpy(edge_mipoint(self.mesh, heh_i)).float().to('cuda')):  
                            for j, heh_j in enumerate(self.mesh.fh(fh)):
                                if self.problem.is_dirichlet(torch.from_numpy(edge_mipoint(self.mesh, heh_j)).float().to('cuda')):
                                    gD = self.problem.dirichlet(torch.from_numpy(edge_mipoint(self.mesh, heh_j)).float().to('cuda'))
                                    self.b[k] += (AK[i, j] * gD)
                    else:
                        fh_Li = adjacent_face(self.mesh, heh_i)
                        li = self.mesh.face_property('global_index', fh_Li)
                        for j, heh_j in enumerate(self.mesh.fh(fh)):
                            if self.problem.is_dirichlet(torch.from_numpy(edge_mipoint(self.mesh, heh_j)).float().to('cuda')):
                                gD = self.problem.dirichlet(torch.from_numpy(edge_mipoint(self.mesh, heh_j)).float().to('cuda'))
                                self.b[k] += (self.mu_K * AK[i, j] * gD)
                                self.b[li]-= (self.mu_L * AK[i, j] * gD)

    def assemble_source_term(self):
        for fh in self.mesh.faces():
            K = torch.from_numpy(self.mesh.calc_face_centroid(fh)[:2]).float().to('cuda')
            fK = self.problem.source(K)
            area_K = area(self.mesh, fh)
            k = self.mesh.face_property('global_index', fh)
            self.b[k] += (fK * area_K) 

    def get_A(self, problem):
        self.A = torch.zeros((self.n_faces, self.n_faces))
        self.problem = problem
        self.assemble_stiffness_matrix()
        return self.A
    
    def get_b(self, problem):
        self.b *= 0
        self.problem = problem
        
        self.treat_boundary_condition()
        # self.assemble_source_term()
        return torch.clone(self.b)
    
    def solve(self, problem=None, tol=1e-13, maxiter=5000):
        
        self.reboot(problem)
        self.assemble_stiffness_matrix()
        self.treat_boundary_condition()
        self.assemble_source_term()

        # Transfer full_matrix to csr_matrix when solving the linear system
        # A = csr_matrix(self.A)
        # b = self.b
        
        # match solver_name:
        #     case 'gmres':            
        #         self.ua, info = gmres(A, b, tol=tol, maxiter=maxiter)
            
        #     case 'bicgstab':
        #         self.ua, info = bicgstab(A, b, tol=tol, maxiter=maxiter)
        
        #     case _:
        self.ua = torch.linalg.solve(self.A, self.b)
        #         info = 0
        
        # if info > 0:
        #     print(f'Convergence to tolerance not achieved! (#it = {info})')
        # elif info < 0:
        #     print('Illegal input or breakdown!')
        
        return self.ua
        