#!/usr/bin/env python
import pyvista as pv
from options import Options

mesh = pv.read("triangle_kershaw_half_1.obj")
mesh.plot(background='white', cpos='xy', show_edges=True, color=True)
#
# def show_mesh(args):
#     mesh = pv.read(args.src)
#     mesh.plot(background='white', cpos='xy', show_edges=True, color=True)
#
# if __name__ == '__main__':
#     args = Options().parse()
#     show_mesh(args)
    
