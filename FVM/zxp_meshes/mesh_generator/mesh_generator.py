#!/usr/bin/env python
import numpy as np
from random import random

def triangle_mesh(m, n, perturb=0, a=1, b=1):
    """Genrate triangle mesh on domain [0,a]x[0,b]
    """
    indices = np.zeros((n+1, m+1)).astype(np.int32)
    count = 0
    hx, hy = 1/m, 1/n

    if perturb == 0:
        if a == 1:
            fname = f'triangle_uniform_{m}x{n}.obj'
        else:
            fname = f'triangle_uniform_half_{m}x{n}.obj'
    else:
        if a == 1:
            fname = f'triangle_random_{m}x{n}.obj'
        else:
            fname = f'triangle_random_half_{m}x{n}.obj'
            
    with open(fname, 'w') as f:
        f.write('# OBJ file\n')
        for j in range(n+1):
            for i in range(m+1):
                rx = random()
                ry = random()
                if 0 < j < n and 0 < i < m:
                    x = i * hx + 2 * (rx - 1) * hx * perturb
                    y = j * hy + 2 * (ry - 1) * hx * perturb
                else:
                    x = i * hx
                    y = j * hy

                x *= a
                y *= b
                f.write(f'v {x} {y} {0.}\n')                
                count += 1
                indices[j, i] = count

        for j in range(n):
            for i in range(m):
                v1 = indices[j, i] 
                v2 = indices[j, i+1] 
                v3 = indices[j+1, i+1] 
                v4 = indices[j+1, i] 
                f.write(f'f {v1} {v2} {v4}\n')
                f.write(f'f {v4} {v2} {v3}\n')

def quadrangle_mesh(m, n, perturb=0, dc=None, a=1, b=1):
    """Genrate quadrangle mesh on domain [0,a]x[0,b]
    """
    indices = np.zeros((n+1, m+1)).astype(np.int32)
    count = 0
    hx, hy = 1/m, 1/n

    if perturb == 0:
        if a == 1:
            fname = f'quadrangle_uniform_{m}x{n}.obj'
        else:
            fname = f'quadrangle_uniform_half_{m}x{n}.obj'
    else:
        if dc is None:
            if a == 1:
                fname = f'quadrangle_random_{m}x{n}.obj'
            else:
                fname = f'quadrangle_random_half_{m}x{n}.obj'                
        if dc == 1:
            if a == 1:
                fname = f'quadrangle_random1_{m}x{n}.obj'
            else:
                fname = f'quadrangle_random1_half_{m}x{n}.obj'
        if dc == 2:
            if a == 1:
                fname = f'quadrangle_random2_{m}x{n}.obj'
            else:
                fname = f'quadrangle_random2_half_{m}x{n}.obj'
        if dc == 4:
            fname = f'quadrangle_random4_{m}x{n}.obj'
            
    with open(fname, 'w') as f:
        f.write('# OBJ file\n')
        for j in range(n+1):
            for i in range(m+1):
                rx = random()
                ry = random()
                if 0 < j < n and 0 < i < m:
                    x = i * hx + 2 * (rx - 1) * hx * perturb
                    y = j * hy + 2 * (ry - 1) * hx * perturb
                    if dc == 1:
                        if i == m//2:
                            x = i * hx
                    if dc == 2:
                        if i == m//2:
                            x = i * hx
                        if j == n//2:
                            y = j * hy
                    if dc == 4:
                        if 3 * (m // 8) <= j <= 5 * (m // 8):
                            if i == 3 * (m // 8) or i == 5 * (m // 8):
                                x = i * hx
                        if 3 * (m // 8) <= i <= 5 * (m // 8):
                            if j == 3 * (m // 8) or j == 5 * (m // 8):
                                y = j * hx                                
                else:
                    x = i * hx
                    y = j * hy

                x *= a
                y *= b
                f.write(f'v {x} {y} {0.}\n')                
                count += 1
                indices[j, i] = count

        for j in range(n):
            for i in range(m):
                v1 = indices[j, i] 
                v2 = indices[j, i+1] 
                v3 = indices[j+1, i+1] 
                v4 = indices[j+1, i] 
                f.write(f'f {v1} {v2} {v3} {v4}\n')

def trapezoidal_mesh(m, n, perturb=0, a=1, b=1):
    indices = np.zeros((n+1, m+1)).astype(np.int32)
    x = np.zeros((n+1, m+1))
    y = np.zeros((n+1, m+1))
    
    count = 0
    hx, hy = 1/m, 1/n
    if a == 1:        
        fname = f'quadrangle_trapezoidal_{m}x{n}.obj'
    else:
        fname = f'quadrangle_trapezoidal_half_{m}x{n}.obj'

    with open(fname, 'w') as f:
        f.write('# OBJ file\n')
        for j in range(n+1):
            for i in range(m+1):
                x[j, i] = i * hx
                y[j, i] = j * hy
                count += 1
                indices[j, i] = count
                
        for j in range(n+1):
            if j % 2 == 1:
                for i in range(n+1):
                    if i % 2 == 1:
                        y[j, i] += hy * perturb
                    else:
                        y[j, i] -= hy * perturb

        for j in range(n+1):
            for i in range(m+1):
                x[j, i] *= a
                y[j, i] *= b
                f.write(f'v {x[j, i]} {y[j, i]} {0.}\n')                
                    

        for j in range(n):
            for i in range(m):
                v1 = indices[j, i] 
                v2 = indices[j, i+1] 
                v3 = indices[j+1, i+1] 
                v4 = indices[j+1, i] 
                f.write(f'f {v1} {v2} {v3} {v4}\n')    
    
def quadrangle_kershaw_mesh(m, n, perturb, a=1, b=1):
    indices = np.zeros((n+1, m+1)).astype(np.int32)
    x = np.zeros((n+1, m+1))
    y = np.zeros((n+1, m+1))
    
    count = 0
    hx, hy = 1/m, 1/n

    if a == 1:
        fname = f'quadrangle_kershaw_{m}x{n}.obj'
    else:
        fname = f'quadrangle_kershaw_half_{m}x{n}.obj'
        
    if m < 8:
        m1 = 2
        m2 = 4
        m3 = m2
        m4 = m - m1
    else:
        m1 = m // 8 
        m2 = m // 4 
        m4 = m - m1
        m3 = m4 - m2 + m1

    with open(fname, 'w') as f:
        f.write('# OBJ file\n')        
        for i in range(m+1):
            if i <= m1:
                hy1 =  perturb
            elif i <= m2: 
                hy1 =  (perturb + (1-2*perturb) * (i-m1) / (m2-m1))
            elif i <= m3:
                hy1 =  (1-perturb + (2*perturb-1) * (i-m2) / (m3-m2))
            elif i <= m4:
                hy1 =  (perturb + (1-2*perturb) * (i-m3) / (m4-m3))
            else:
                hy1 =  (1-perturb)

            hy2 = 1 - hy1
            hy1  /= m//2
            if m // 2 == 0:
                hy2 /= m//2
            else:
                hy2 /= (m+1)//2
            
            for j in range(n+1):
                x[j, i] = i * hx
                if j <= n//2:
                    y[j, i] = j * hy1
                else:
                    y[j, i] = 1 -  (n - j) * hy2
            
            
        for j in range(n+1):
            for i in range(m+1):
                x[j, i] *= a
                y[j, i] *= b
                f.write(f'v {x[j, i]} {y[j, i]} {0.}\n')
                count += 1
                indices[j, i] = count
                    

        for j in range(n):
            for i in range(m):
                v1 = indices[j, i] 
                v2 = indices[j, i+1] 
                v3 = indices[j+1, i+1] 
                v4 = indices[j+1, i] 
                f.write(f'f {v1} {v2} {v3} {v4}\n')   

def triangle_kershaw_mesh(m, n, perturb, a=1, b=1):
    indices = np.zeros((n+1, m+1)).astype(np.int32)
    x = np.zeros((n+1, m+1))
    y = np.zeros((n+1, m+1))
    
    count = 0
    hx, hy = 1/m, 1/n
    
    if a == 1:
        fname = f'triangle_kershaw_{m}x{n}.obj'
    else:
        fname = f'triangle_kershaw_half_{m}x{n}.obj'
        
    if m < 8:
        m1 = 2
        m2 = 4
        m3 = m2
        m4 = m - m1
    else:
        m1 = m // 8 
        m2 = m // 4 
        m4 = m - m1
        m3 = m4 - m2 + m1

    with open(fname, 'w') as f:
        f.write('# OBJ file\n')        
        for i in range(m+1):
            if i <= m1:
                hy1 =  perturb
            elif i <= m2: 
                hy1 =  (perturb + (1-2*perturb) * (i-m1) / (m2-m1))
            elif i <= m3:
                hy1 =  (1-perturb + (2*perturb-1) * (i-m2) / (m3-m2))
            elif i <= m4:
                hy1 =  (perturb + (1-2*perturb) * (i-m3) / (m4-m3))
            else:
                hy1 =  (1-perturb)

            hy2 = 1 - hy1
            hy1  /= m//2
            if m // 2 == 0:
                hy2 /= m//2
            else:
                hy2 /= (m+1)//2
            
            for j in range(n+1):
                x[j, i] = i * hx
                if j <= n//2:
                    y[j, i] = j * hy1
                else:
                    y[j, i] = 1 -  (n - j) * hy2
            
            
        for j in range(n+1):
            for i in range(m+1):
                x[j, i] *= a
                y[j, i] *= b
                f.write(f'v {x[j, i]} {y[j, i]} {0.}\n')
                count += 1
                indices[j, i] = count
                    

        for j in range(n):
            for i in range(m):
                v1 = indices[j, i] 
                v2 = indices[j, i+1] 
                v3 = indices[j+1, i+1] 
                v4 = indices[j+1, i] 
                f.write(f'f {v1} {v2} {v3}\n')
                f.write(f'f {v1} {v3} {v4}\n')

                
def shashkov_mesh(m, n, perturb, a=1, b=1):
    """Generator of Shashkov mesh, and perturb is perferrable in (0, 0.1)"""
    indices = np.zeros((n+1, m+1)).astype(np.int32)
    x = np.zeros((n+1, m+1))
    y = np.zeros((n+1, m+1))
    
    count = 0
    hx, hy = 1/m, 1/n
    
    if a==1:
        fname = f'quadrangle_shashkov_{m}x{n}.obj'
    else:
        fname = f'quadrangle_shashkov_half_{m}x{n}.obj'
        
    with open(fname, 'w') as f:
        f.write('# OBJ file\n')
        for j in range(n+1):
            for i in range(m+1):                
                x[j, i] = i * hx
                y[j, i] = j * hy
                count += 1
                indices[j, i] = count

        for j in range(n+1):
            for i in range(m+1):                
                t = perturb * np.sin(2*np.pi*x[j, i]) * np.sin(2*np.pi*y[j, i])
                x[j, i] += t
                y[j, i] += t
    
        for j in range(n+1):
            for i in range(m+1):
                f.write(f'v {x[j, i]} {y[j, i]} {0.}\n')                
                    

        for j in range(n):
            for i in range(m):
                v1 = indices[j, i] 
                v2 = indices[j, i+1] 
                v3 = indices[j+1, i+1] 
                v4 = indices[j+1, i] 
                f.write(f'f {v1} {v2} {v3} {v4}\n')    
   
if __name__ == '__main__':

    # for m in [8, 16, 32, 64, 128]:
    #     triangle_mesh(m, m, 0)
        
    # for m in [8, 16, 32, 64, 128]:
    #     triangle_mesh(m, m, 0.3)
          # triangle_mesh(m, m, 0.3, a=0.5, b=0.5)
        
    # for m in [8, 16, 32, 64, 128]:
    #     quadrangle_mesh(m, m, 0)

    # for m in [8, 16, 32, 64, 128]:
        # quadrangle_mesh(m, m, 0.3)
        # quadrangle_mesh(m, m, 0.3, a=0.5, b=0.5)

    for m in [8, 16, 32, 64, 128]:
        quadrangle_mesh(m, m, 0.3, dc=4)
    # for m in [4, 8, 16, 32, 64]:
    #     quadrangle_mesh(m, m, 0.3, dc=1)        
        
    # for m in [4, 8, 16, 32, 64, 128]:
    #     quadrangle_mesh(m, m, 0.3, dc=2)        
        
    # for m in [4, 8, 16, 32, 64]:
    #     trapezoidal_mesh(m, m, 0.5)

    # for m in [8, 16, 32, 64, 128]:
    #     # quadrangle_kershaw_mesh(m, m, 0.2)
    #     quadrangle_kershaw_mesh(m, m, 0.2, a=0.5, b=0.5)

    # for m in [8, 16, 32, 64, 128]:
    #     # triangle_kershaw_mesh(m, m, 0.2)
    #     triangle_kershaw_mesh(m, m, 0.2, a=0.5, b=0.5)
        
    # for m in [4, 8, 16, 32, 64]:
    #     shashkov_mesh(m, m, 0.1)

    
