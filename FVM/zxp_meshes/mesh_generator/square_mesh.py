#!/usr/bin/env python
import numpy as np

def generate_square_mesh(m, n):

    indices = {}
    count = 0
    hx, hy = 1/m, 1/n
    with open(f'square_{m}_{n}.obj', 'w') as f:
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

                
if __name__ == '__main__':
    generate_square_mesh(64, 64)
