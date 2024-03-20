#!/usr/bin/env python3
import pyvista as pv
from options import Options

def show_mesh(args):
    # pv.global_theme.lighting = True
    mesh = pv.read(args.src)
    mesh['scalars'] = mesh.points[:, 2]
    mesh.plot(background='white', cpos='xy', cmap='plasma', show_scalar_bar=False)
    mesh.plot(background='w', cpos='xy', show_edges=True, edge_color='k', color='c', cmap='plasma')
    p = pv.Plotter(off_screen=True)
    p.add_mesh(mesh, show_edges=True, color=True)
    p.remove_bounding_box()
    p.show()
    # plotter.show(screenshot='airplane.png', cpos='xy')

    # mesh = pv.read(args.src)
    # mesh['scalars'] = mesh.points[:, 2]
    # contours = mesh.contour(isosurfaces=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # p = pv.Plotter()
    # p.add_mesh(mesh, opacity=0.85)
    # p.add_mesh(contours, color="white", line_width=5)    
    # p.view_xy()
    # p.show()
    
if __name__ == '__main__':
    pv.read('./quadrangle_kershaw_1.obj')
    # args = Options().parse()
    # show_mesh(args)
    
