#!/usr/bin/env python
import numpy as np

def triangular_mesh(m, n, perturb=0):

    indices = {}
    count = 0
    hx, hy = 1/m, 1/n

    if perturb == 0:
        fname = f'triangular_regular_{m}x{n}.obj'
    else:
        fname = f'triangular_random_{m}x{n}.obj'
        
    with open(fname, 'w') as f:
        f.write('# OBJ file\n')
        for j in range(n+1):
            for i in range(m+1):
                f.write(f'v {i*hx} {j*hy} {0.}\n')                
                count += 1
                indices[(j,i)] = count

        for j in range(n):
            for i in range(m):
                v1 = indices[(j, i)] 
                v2 = indices[(j, i+1)] 
                v3 = indices[(j+1, i+1)] 
                v4 = indices[(j+1, i)] 
                f.write(f'f {v1} {v2} {v4}\n')
                f.write(f'f {v4} {v2} {v3}\n')

def kershaw_mesh():
    pass 
                
if __name__ == '__main__':
    triangular_mesh(8, 8)
