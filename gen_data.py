import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from FVM.src.utils import readmesh, ChipLayout, PieceWiseConst
from FVM.src.ICD import VolumnCenteredScheme
from FVM.src.Problem import *
from scipy import sparse
from scipy.sparse.linalg import spsolve

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
        
        mesh = readmesh(f'./FVM/my_meshes/UniformQuad-HeatChip-{GridSize}.obj')
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

            norm_problem = NormChipHeatDissipation(None, case, eps=h**2)
            A = solver.get_A(norm_problem).tocsr()
            b = solver.get_b(norm_problem)
            sparse.save_npz(save_path/'normA.npz', A)
            np.save(save_path/'normb.npy', b)

            if solve:
                B = Force.reshape(self.dataN, -1) * h**2 + b[np.newaxis, ...]
                U = spsolve(A, B.transpose()).transpose().reshape((self.dataN, GridSize, GridSize))
                np.save(save_path/'normU.npy', U)
        return 

class CofsDataGenerator:
    def __init__(self, DataN, minK, maxK, ):
        self.DataN = DataN
        self.minK = minK
        self.maxK = maxK
            
    def cofs2csv(self, t, csv_save_path=None):
        infos = np.random.uniform(self.minK, self.maxK, (self.DataN, t**2))
        if isinstance(infos, list):
            infos = np.stack(infos, axis=0)
        cols = {f"a_{i}": [] for i in range(t**2)}
        df = pd.DataFrame(infos, columns = cols)
        if not csv_save_path is None:   
            df.to_csv(csv_save_path, index=False)
        return df

    def cofs_assemble(self, df, t, GridSize):
        mus = df.values  
        mesh = readmesh(f'./FVM/my_meshes/UniformQuad-VaryK-{GridSize}.obj')
        
        # Save A b first
        solver = VolumnCenteredScheme(mesh)
        for i, mu in tqdm(enumerate(mus)):
            mu = mu.reshape(t, t)
            problem = VaryDiffusionCof(mu)
            A = solver.get_A(problem).tocsr()
            sparse.save_npz(f'{self.data_path}/A{i}.npz', A)
        
        b = solver.get_b(problem)
        np.save(f'{self.data_path}/b.npy', b)

    def cofs_solve(self, GridSize):
        b = np.load(f'{self.data_path}/b.npy')
        U = []
        for i in range(self.DataN):
            A = sparse.load_npz(f'{self.data_path}/A{i}.npz')
            sol = spsolve(A, b).reshape(GridSize, GridSize)
            U.append(sol)
        U = np.stack(U, axis=0)
        np.save(f'{self.data_path}/U.npy', U)
    
    def generate(self, t, csv_save_path, data_path, GridSize):
        data_path = f"{data_path}/square{t}x{t}/GridSize-{GridSize}"
        if not Path(self.data_path).is_dir():
            Path(self.data_path).mkdir(parents=True)

        df = self.cofs2csv(t, csv_save_path)
        self.cofs_assemble(df, t, GridSize)
        self.cofs_solve(GridSize)

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

def gen_one_nlinear_data(GridSize, area, mu, Picard_maxiter=1000):
    (left, bottom), (right, top) = area
    h = (right - left) / GridSize

    mesh = readmesh(f'./FVM/my_meshes/UniformQuad-VaryK-{GridSize}.obj')
    solver = VolumnCenteredScheme(mesh)

    problem = NLinearProblem(h, None, (0.5, 0.5), area, mu)
    solver.solve(problem, solver_name = None)
    u0, A0, b = solver.ua, solver.A, solver.b

    for _ in range(Picard_maxiter):
        new_problem = NLinearProblem(h, u0.reshape(GridSize, GridSize), (0.5, 0.5), area, mu)
        newA = solver.get_A(new_problem).tocsr()
        newu = spsolve(newA, b)

        delta = ((newu - u0)**2 * h**2).sum()
        error = np.linalg.norm(A0 @ newu - b)
        if delta < 1e-6 or error < 1e-6:
            u = newu.reshape(GridSize, GridSize)
            break
        else:
            u0, A0 = newu, newA
    
    return u


if __name__ == '__main__':
    path = './DLdata'
    DataN = 10000
    GridSize=128

    # #  Heat and NormHeat Data
    # heat_generator = ChipsDataGenerator(DataN, ((0, 0), (0.1, 0.1)), 0.001, 0.001)
    # heat_generator.generate(f'{path}/heat_info.csv', f'{path}/heat', 32, True)

    # # VaryCof data
    # for t in [2, 3, 4]:
    #     varyk_generator = CofsDataGenerator(t, DataN, 0.1, 10, f'{path}/varyk', 32)
    #     varyk_generator.generate(f'{path}/square{t}x{t}.csv')

    # # Water Dataset
    water_generator = WaterDataGenerator(DataN, ((-250, -250), (250, 250)), 25, 25, 50, 150, 8)
    # water_generator.generate(f'{path}/water_info.csv', f'{path}/water', GridSize, False, True)
    water_generator.generate(f'{path}/hetero-water_info.csv', f'{path}/hetero-water', GridSize, True, True)

    