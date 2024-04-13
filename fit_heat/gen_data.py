import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from FVM.src.utils import readmesh, ChipLayout, PieceWiseConst
from FVM.src.ICD import VolumnCenteredScheme
from FVM.src.Problem import *
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from pathlib import Path

chips = [
    [0.016, 0.012, 4000], [0.012, 0.006, 16000], [0.018, 0.009, 6000], [0.018, 0.012, 8000],
    [0.018, 0.018, 10000], [0.012, 0.012, 14000],[0.018, 0.006, 16000], [0.009, 0.009, 20000],
    [0.006, 0.024, 8000], [0.006, 0.012, 16000], [0.012, 0.024, 10000], [0.024, 0.024, 20000]]

def _mesh(area, GridSize):
    x0, y0 = area[0]
    x1, y1 = area[1]
    dx = (x1 - x0) / GridSize
    dy = (y1 - y0) / GridSize
    x = np.arange(x0 + dx/2, x1, dx)
    y = np.arange(y0 + dy/2, y1, dy)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

class ChipsDataGenerator:
    def __init__(self, DataN, area, boundary_gap, chip_gap):
        self.chips = chips
        self.DataN = DataN
        self.area = area
        self.boundary_gap = boundary_gap
        self.chip_gap = chip_gap

    def layout2csv(self, csv_save_path):
        infos = []
        pbar = tqdm(total = self.DataN)
        while len(infos) < self.DataN:
            try:
                i = len(infos) + 1
                info = self.SeqLS(i)
            except:
                continue
            infos.append(info)
            pbar.update(1)

        infos = np.vstack(infos)
        if csv_save_path:
            dic = {
                'idx': 'int',
                'x':'float', 
                'y':'float', 
                'w':'float', 
                'h':'float', 
                'c':'float'
            }
            df = pd.DataFrame(infos, columns=dic.keys()).astype(dic)
            df.to_csv(csv_save_path, index=False)
        return df

    def SeqLS(self, idx, GridSize=500):
        (left, bottom), (right, top) = self.area
        vx = np.linspace(left , right , GridSize)
        vy = np.linspace(bottom , top, GridSize)
        xx, yy = np.meshgrid(vx, vy)
        info = []
        for i, chip in enumerate(chips):
            w, h, c = chip
            def _func0(x, y):
                sign_x = (x >= left + self.boundary_gap) & (x < right - self.boundary_gap - w)
                sign_y = (y >= bottom + self.boundary_gap) & (y < top - self.boundary_gap - h)
                sign = sign_x & sign_y
                return ~sign
            eVEMs = []
            eVEMs.append(_func0(xx, yy))

            for j in range(i):
                _, xj, yj, wj, hj, _ = info[j]
                l = xj - w - self.chip_gap 
                r = xj + wj + self.chip_gap 
                b = yj - h - self.chip_gap 
                t = yj + hj + self.chip_gap 

                def _funcj(x, y):
                    sign_x = (x >= l) & (x < r)
                    sign_y = (y >= b) & (y < t)
                    sign = sign_x & sign_y
                    return sign
                
                eVEMs.append(_funcj(xx, yy))
            IeVEM = np.sum(np.stack(eVEMs, axis=0), axis=0)
            legal_indices = np.argwhere(IeVEM == 0)
            if len(legal_indices) > 0:
                idx_x, idx_y = legal_indices[np.random.choice(len(legal_indices))]
                info.append((idx, vx[idx_y], vy[idx_x], w, h, c))
            else:
                raise ValueError
        return info

    def generate(self, csv_save_path, data_path, GridSize=128, solve=True):
        df = self.layout2csv(csv_save_path)
        h = self.area[-1][-1] / GridSize
        xx, yy = _mesh(self.area, GridSize)
        
        layouts = []
        for _, data in df.groupby('idx'):
            info = data.values[:, 1:]
            layouts.append(ChipLayout(info))
        
        mesh = readmesh(f'../FVM/my_meshes/UniformQuad-HeatChip-{GridSize}.obj')
        solver = VolumnCenteredScheme(mesh=mesh)

        Force = np.stack([f(xx, yy) for f in layouts], axis=0)
        F_save_path = Path(f'{data_path}/GridSize-{GridSize}')
        if not F_save_path.is_dir():
            F_save_path.mkdir(parents=True)
        np.save(F_save_path/'F.npy', Force)

        for case in [1, 2, 3]:
            save_path = Path(f'{data_path}/GridSize-{GridSize}/case-{case}')
            if not save_path.is_dir():
                save_path.mkdir(parents=True)

            problem = ChipHeatDissipation(None, case, eps=h**2)
            A = solver.get_A(problem).tocsr()
            b = solver.get_b(problem)
            sparse.save_npz(save_path/'A.npz', A)
            np.save(save_path/'b.npy', b)
            if solve:
                B = Force.reshape(self.dataN, -1) * h**2 + b[np.newaxis, ...]
                U = spsolve(A, B.transpose()).transpose().reshape((self.dataN, GridSize, GridSize))
                np.save(save_path/'U.npy', U)        

            # norm_problem = NormChipHeatDissipation(None, case, eps=h**2)
            # A = solver.get_A(norm_problem).tocsr()
            # b = solver.get_b(norm_problem)
            # sparse.save_npz(save_path/'normA.npz', A)
            # np.save(save_path/'normb.npy', b)

            # if solve:
            #     B = Force.reshape(self.dataN, -1) * h**2 + b[np.newaxis, ...]
            #     U = spsolve(A, B.transpose()).transpose().reshape((self.dataN, GridSize, GridSize))
            #     np.save(save_path/'normU.npy', U)
        return 

if __name__ == '__main__':
    DataN = 10
    area = ((0, 0), (0.1, 0.1))
    boundary_gap = 0.01
    chip_gap = 0.01

    generator = ChipsDataGenerator(DataN, area, boundary_gap, chip_gap)
    generator.generate(
        './DLdata/info.csv',
        './DLdata',
        64,
        False
    )