import openmesh as om
import numpy as np
from scipy.spatial.distance import pdist

def show_errors(hmeshes, errors_max, errors_l2, errors_h1):
    for i in range(len(hmeshes)):
        if i == 0:
            s = f"{hmeshes[i]:11.3e}|" + \
                f"{errors_max[i]:11.3e}|{' ':7s}|" + \
                f"{errors_l2[i]:11.3e}|{' ':7s}|" + \
                f"{errors_h1[i]:11.3e}|{' ':7s}|"
        else:
            r_max = (np.log(errors_max[i-1]) - np.log(errors_max[i])) / (np.log(hmeshes[i-1]) - np.log(hmeshes[i]))
            r_l2  = (np.log(errors_l2[i-1] ) - np.log(errors_l2[i] )) / (np.log(hmeshes[i-1]) - np.log(hmeshes[i]))
            r_h1  = (np.log(errors_h1[i-1] ) - np.log(errors_h1[i] )) / (np.log(hmeshes[i-1]) - np.log(hmeshes[i]))
            s = f'{hmeshes[i]:11.3e}|'+\
                f'{errors_max[i]:11.3e}|{r_max:7.3f}|'+\
                f'{errors_l2[i]:11.3e}|{r_l2:7.3f}|'+\
                f'{errors_h1[i]:11.3e}|{r_h1:7.3f}|'
        yield s  

# Generate a Uniformly quadrangle mesh
def uniform_quad_mesh(left_bottom, right_top, Nx, Ny,  filename):
    x0, y0 = left_bottom
    x1, y1 = right_top
    indices = np.zeros((Nx, Ny), np.int32)
    count = 0

    dx =  (x1 - x0) / (Nx-1)
    dy =  (y1 - y0) / (Ny-1)

    with open(filename, 'w') as f:
        # Indiced by row, from the first row to the last row
        # add vertices
        for i in range(Ny):
            for j in range(Nx):
                x = x0 + j * dx
                y = y0 + i * dy 

                f.write(f'v {x} {y} {0.}\n')
                count += 1
                indices[i, j] = count
        
        # add faces Counterclock
        for i in range(Ny-1):
            for j in range(Nx-1):
                v1 = indices[i, j]
                v2 = indices[i, j+1]
                v3 = indices[i+1, j+1]
                v4 = indices[i+1, j]                
                f.write(f'f {v1} {v2} {v3} {v4}\n')
    return True

def uniform_tri_mesh(left_bottom, right_top, Nx, Ny,  filename):
    x0, y0 = left_bottom
    x1, y1 = right_top
    indices = np.zeros((Nx, Ny), np.int32)
    count = 0

    dx =  (x1 - x0) / (Nx-1)
    dy =  (y1 - y0) / (Ny-1)

    with open(filename, 'w') as f:
        # Indiced by row, from the first row to the last row
        # add vertices
        for i in range(Ny):
            for j in range(Nx):
                x = x0 + j * dx
                y = y0 + i * dy 

                f.write(f'v {x} {y} {0.}\n')
                count += 1
                indices[i, j] = count
        
        for i in range(Ny-1):
            for j in range(Nx-1):
                v1 = indices[i, j]
                v2 = indices[i, j+1]
                v3 = indices[i+1, j]
                f.write(f'f {v1} {v2} {v3}\n')

                v1 = indices[i+1, j+1]
                v2 = indices[i+1, j]
                v3 = indices[i, j+1]                
                f.write(f'f {v1} {v2} {v3}\n')
    return True

def readmesh(fname):
    """read mesh and then processing it"""
    mesh = om.read_polymesh(fname)

    if not mesh.has_vertex_property('global_index'):
        mesh.vertex_property('global_index')            
        
    if not mesh.has_face_property('global_index'):
        mesh.face_property('global_index')
        
    if not mesh.has_face_property('diameter'):
        mesh.face_property('diameter')

    # Set global index
    index = 0
    for fh in mesh.faces():
        mesh.set_face_property('global_index', fh, index)
        index += 1

    for vh in mesh.vertices():
        mesh.set_vertex_property('global_index', vh, index)
        index += 1

    # Set diameter
    for fh in mesh.faces():
        vertices = np.vstack([mesh.point(vh)[:2] for vh in mesh.fv(fh)])
        diameter = np.max(pdist(vertices))
        mesh.set_face_property('diameter', fh, diameter)
        
    return mesh

def prev_vertex(mesh, vh, fh):
    for heh in mesh.fh(fh):
        if mesh.to_vertex_handle(heh) == vh:
            return mesh.to_vertex_handle(heh)
    
def next_vertex(mesh, vh, fh):
    for heh in mesh.fh(fh):
        if mesh.from_vertex_handle(heh) == vh:
            return mesh.to_vertex_handle(heh)
    
def prev_halfedge(mesh, heh):
    return mesh.prev_halfedge_handle(heh)
        
def next_halfedge(mesh, heh):
    return mesh.next_halfedge_handle(heh)

def adjacent_halfedge(mesh, heh):
    return mesh.opposite_halfedge_handle(heh)

def adjacent_face(mesh, heh):
    op_heh =  adjacent_halfedge(mesh, heh)
    if is_boundary(mesh, op_heh):
        return None
    else:
        return mesh.face_handle(op_heh)

def fh_n_edges(mesh, fh):
    return len(list(mesh.fh(fh)))

def area(mesh, fh):
    vertices = [mesh.point(vh) for vh in mesh.fv(fh)]
    N = len(vertices)
    o = mesh.calc_face_centroid(fh)
    triangles = []
    for i in range(N):
        triangles.append(np.stack([o, vertices[i], vertices[(i+1)%N]]))
    triangles = np.stack(triangles)
    areas = np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        axis = 1), axis=1) / 2
    return areas.sum() 

def adjacent_center(mesh, heh):
    adjacent_fh = adjacent_face(mesh, heh)
    if adjacent_fh is None:
        return edge_mipoint(mesh, heh)
    else:
        v = mesh.calc_face_centroid(adjacent_fh)
        return v[:2]
    
def is_boundary(mesh, heh):
    return mesh.is_boundary(mesh.edge_handle(heh))

def edge_mipoint(mesh, heh):
    vh1 = mesh.point(mesh.from_vertex_handle(heh))[:2]
    vh2 = mesh.point(mesh.to_vertex_handle(heh))[:2]
    sigma = (vh1 + vh2) / 2
    return sigma

def tangent_vector(mesh, heh):
    vec = mesh.calc_edge_vector(heh)[:2]
    vec = vec / np.linalg.norm(vec)
    return vec

def normal_vector(mesh, heh):
    vec_tangent = tangent_vector(mesh, heh)
    vec_normal = np.array([[0, 1], [-1, 0]]) @ vec_tangent
    return vec_normal

class ChipLayout: 
    def __init__(self, info):
        self.info = info
    
    def __call__(self, X, Y):
        heat = np.zeros_like(X)
        for xi, yi, wi, hi, ci in self.info:
            sign_x = (X >= xi) & (X < xi + wi)
            sign_y = (Y >= yi) & (Y < yi + hi)
            sign = sign_x & sign_y
            heat += (ci * sign)
        return heat
    
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

if __name__ == '__main__':
    for N in [32, 64, 128, 256, 512]:
        uniform_quad_mesh((-250, -250), (250, 250), N+1, N+1, filename=f'../my_meshes/UniformQuad-WaterFlow-{N}.obj')
    pass