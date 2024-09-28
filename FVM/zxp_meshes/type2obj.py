#!/usr/bin/env python
import numpy as np
import pandas as pd
from options import Options

def type2obj(args):
    with open(args.src) as f:
        for line in f:
            line = line.strip().split()                
            if line[0] == "Vertices":
                vertices = []
                i = 0
                type = 'Vertices'
            elif line[0] == "cells":
                cells = []
                i = 0
                type = 'cells'
            elif len(line) == 1:
                if type == 'Vertices':
                    n_vertices = int(line[0])
                else:
                    n_cells = int(line[0])
                    i = 0
            elif type == 'Vertices' and i < n_vertices:
                vertices.append(np.array(line).astype(np.float64))
                i += 1
            elif type == 'cells' and i < n_cells:
                cells.append(np.array(line).astype(np.int32))
                i += 1

    with open(args.des, 'w') as f:
        f.write('# OBJ file\n')
        for vertice in vertices:
            f.write(f'v {vertice[0]:.10f} {vertice[1]:.10f} {0:.10f}\n')
        for cell in cells:
            f.write(f'f {str(cell[1:])[1:-1]}\n')

if __name__ == '__main__':
    args = Options().parse()    
    type2obj(args)
    
