from src.Problem import zxpProblem
from src.ICD import VolumnCenteredScheme
from src.utils import show_errors, readmesh

# Test zxp case !!!
case_ns = list(range(10))
print(f"|{'Method': ^8s}|{'Case ID': ^8s}|{'Mesh Type': ^18s}|{'hmesh': ^11s}|{'error_max': ^11s}|{'ratio': ^7s}|{'error_l2': ^11s}|{'ratio': ^7s}|{'error_h1': ^11s}|{'ratio': ^7s}|")
print(f'|{":--:": ^8s}|{":--:": ^8s}|{":--:": ^18s}|{":--:": ^11s}|{":--:": ^11s}|{":--:": ^7s}|{":--:": ^11s}|{":--:": ^7s}|{":--:": ^11s}|{":--:": ^7s}|')
for case_n in case_ns:
    mesh_type = 'triangle_uniform'
        # triangle_classic triangle_random triangle_uniform triangle_kershaw
        # quadrangle_uniform quadrangle_random  quadrangle_kershaw
    n_levels = 5
    hmeshes = []
    errors_max = []
    errors_l2 = []
    errors_h1 = []

    for i in range(n_levels):
        fname = f'./zxp_meshes/{mesh_type}_{i+1}.obj'
        mesh = readmesh(fname)
        problem = zxpProblem(case_n, 1e-9)

        solver = VolumnCenteredScheme(mesh, problem)
        solver.solve(solver_name=None)
        hmesh, error_max, error_l2, error_h1 = solver.compute_errors()

        hmeshes.append(hmesh)
        errors_max.append(error_max)
        errors_l2.append(error_l2)
        errors_h1.append(error_h1)

    for s in show_errors(hmeshes, errors_max, errors_l2, errors_h1):
        print(f'|{"ICD": ^8s}|{case_n: ^8}|{mesh_type: ^18s}|' + s)
