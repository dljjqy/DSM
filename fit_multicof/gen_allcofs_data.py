import sys
sys.path.append('../')

from scipy.sparse import save_npz
from pathlib import Path
from utils import PieceWiseConst
from ConstCofFVM import *
from UniformICD import UniformFVM
from Problems import BlockCofProblem
from tqdm import tqdm
import numpy as np

import argparse

# def gen_linsys(idx, cof, save_path, area, GridSize):
#     problem = BlockCofProblem(cof, GridSize, area)
#     solver = UniformFVM(area, GridSize, GridSize, problem)
#     A, b = solver.get_A(problem)
#     save_npz(f'{save_path}/a{idx}.npz', A)
#     if Path(f'{save_path}/b.npy').is_file():
#         np.save(f'{save_path}/b.npy', b)
#     np.save(f'{save_path}/c{idx}.npy', cof)

def gen_data(start, N, GridSize, save_path='./DLdata/ForTest', area=((0, 0), (1, 1))):
    save_path = save_path + f'/{GridSize}'
    p = Path(save_path)
    if not p.is_dir():
        p.mkdir(parents=True)

    cofs = []
    (left, bottom), (right, top) = area
    dx = (right - left) / GridSize
    dy = (top - bottom) / GridSize
    xx, yy = np.meshgrid(
        np.arange(left + dx/2, right, dx),
        np.arange(bottom + dy/2, top, dy)
    )

    for _ in range(N):
        t = np.random.choice([2, 3, 4])
        mu = np.random.uniform(0.1, 10, (t, t))
        pwc = PieceWiseConst(mu, area)
        cofs.append(pwc(xx, yy))

    solver = UniformFVM(area, GridSize, GridSize, None)
    pbar = tqdm(total=N)
    for idx, cof in enumerate(cofs):
        problem = BlockCofProblem(cof, GridSize, area)
        A = solver.get_A(problem)
        b = solver.get_B(problem)
        u = sparse.linalg.spsolve(A, b).reshape(GridSize, GridSize)
        # solver.solve(solver_name=None)
        save_npz(f'{save_path}/a{start + idx}.npz', A)
        np.save(f'{save_path}/b{start + idx}.npy', b)
        np.save(f'{save_path}/c{start + idx}.npy', cof)
        np.save(f'{save_path}/u{start + idx}.npy', u)

        pbar.update(1)


if __name__ == '__main__':
    # start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',
                        type=int,
                        default=0,
                        help='Start Index')
    parser.add_argument('--dataN',
                        type=int,
                        default=1000,
                        help='Start Index')
    parser.add_argument('--GridSize',
                        type=int,
                        default=96,
                        help='Start Index')
    
    args = parser.parse_args()
    # print(args.start)
    gen_data(args.start, args.dataN, args.GridSize, './DLdata/')
    # print(time() - start)
    # print(parser.parse_args())