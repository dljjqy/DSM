import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pathlib import Path

from FVM.src.ICD import VolumnCenteredScheme
from FVM.src.Problem import WaterPump, LinearWaterFlow, HeterWaterFlow
from FVM.src.utils import readmesh

def _mesh(area, GridSize):
    x0, y0 = area[0]
    x1, y1 = area[1]
    dx = (x1 - x0) / GridSize
    dy = (y1 - y0) / GridSize
    x = np.arange(x0 + dx/2, x1, dx)
    y = np.arange(y0 + dy/2, y1, dy)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

class WaterDataGenerator:
    def __init__(self, DataN, area,  sideGap, sourceGap, minQ, maxQ, maxPointN):
        self.DataN = DataN
        self.area = area
        self.sideGap = sideGap
        self.sourceGap = sourceGap
        self.minQ = minQ
        self.maxQ = maxQ
        self.maxPointN = maxPointN
    
    def gen_random_locs(self, PointN, idx):
        left, bottom = self.area[0]
        right, top = self.area[1]
        Qs = np.random.uniform(self.minQ, self.maxQ, PointN)
        locs = []
        pumps = []
        point_nums = 0
        while point_nums < PointN:
            new_loc = (np.random.uniform(left+self.sideGap, right-self.sideGap),
                       np.random.uniform(bottom+self.sideGap, top-self.sideGap))
            if len(locs) == 0:
                pumps.append(np.array([idx, new_loc[0], new_loc[1], Qs[point_nums]]))
                locs.append(new_loc)
                point_nums += 1
                continue
            is_legal = (np.linalg.norm(np.array(locs)[1:2] - new_loc, axis=1) > self.sourceGap).all()
            if is_legal:
                locs.append(new_loc)
                pumps.append(np.array([idx, new_loc[0], new_loc[1], Qs[point_nums]]))

                point_nums += 1
        pumps =  np.stack(pumps, axis=0)
        return pumps
    
    def layout2csv(self, csv_save_path):
        infos = []
        PointNs = np.random.choice(list(range(1, self.maxPointN+1)), self.DataN)
        for idx in tqdm(range(self.DataN)):
            infos.append(
                self.gen_random_locs(PointNs[idx], idx)
            )
        infos = np.concatenate(infos, axis=0)
        if csv_save_path:
            dic = {
                'idx': 'int',
                'x':'float', 
                'y':'float', 
                'Q':'float', 
            }
            df = pd.DataFrame(infos, columns=dic.keys()).astype(dic)
            df.to_csv(csv_save_path)
        return df
    
    def generate(self, csv_save_path, data_path, GridSize, Hetero = False, solve=True):
        df = self.layout2csv(csv_save_path)
        h = (self.area[1][0] - self.area[0][0])/GridSize
        xx, yy = _mesh(self.area, GridSize)
        mesh = readmesh(f'./FVM/my_meshes/UniformQuad-WaterFlow-{GridSize}.obj')
        solver = VolumnCenteredScheme(mesh)
        
        Force = []
        for _, data in df.groupby('idx'):
            locs, Qs = data.values[:, 1:3], data.values[:, 3]
            Func = WaterPump(locs, Qs, h)
            Force.append(Func(xx, yy))
        Force = np.stack(Force, axis=0)
        
        for bd_case in [1, 2]:
            p = Path(f'{data_path}/GridSize-{GridSize}/case{bd_case}')
            if not p.is_dir():
                p.mkdir(parents=True)
            
            if Hetero:
                problem = HeterWaterFlow(None, bd_case, self.area, eps=h**2)
                tag = 'hetero'
            else:
                problem = LinearWaterFlow(None, bd_case, self.area, eps=h**2)
                tag = ''

            A = solver.get_A(problem).tocsr()
            b = solver.get_b(problem)
            
            sparse.save_npz(p/'A.npz', A)
            np.save(p/'b.npy', b)
            np.save(f'{data_path}/GridSize-{GridSize}/F.npy', Force)
            if solve:
                B = 100 * Force.reshape(self.DataN, -1) * h**2 + b
                U = spsolve(A, B.transpose()).transpose().reshape((self.DataN, GridSize, GridSize))
                np.save(p/'U.npy', U)