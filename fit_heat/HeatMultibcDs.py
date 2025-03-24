import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from random import choice
import itertools

import sys
sys.path.append('../')
from utils import ChipLayout

ALL_DATA_NUM = 15000

def case1( x, y):
    mask = ((x == 0) + (y == 0) + (x == 0.1) + (y == 0.1)) > 0
    return mask * 298

def case2( x, y):
    mask = x == 0
    return mask * 298

def case3( x, y):
    mask = (y == 0) * (x >= 0.0495) * (x <= 0.0505)
    return mask * 298

class AbstractClass(Dataset):
    def __init__(self, GridSize, N, dtype, device, batch_size=5, usage='val'):
        super(AbstractClass, self).__init__()
        self.GridSize = GridSize
        self.N = N
        self.dtype = dtype
        self.device = device
        self._get_data_range(usage)

        self.cases = np.random.choice([1,2,3], N)
        match usage:
            case 'val':
                # self.cases = [choice([1, 2, 3]) for _ in range(N)]
                self.U = self._loadU()
            case 'train':
                pass
                # cases = [choice([[1]*batch_size, [2]*batch_size, [3]*batch_size]) for _ in  range(N//batch_size)]
                # self.cases = list(itertools.chain(*cases))
        self.layouts = self._load_layouts()
        self.bd_tensors = self._load_boundary()
        # self.B = self._loadB()

    
    def _to(self, x):
        return x.to(self.dtype).to(self.device)

    def __len__(self):
        return self.N
    
    def _assemble_data(self, index):
        layout = self.layouts[index]
        boundary_tensor = self.bd_tensors[self.cases[index]-1].clone()
        data = torch.vstack([boundary_tensor, layout[None, ...]])
        return data
    
    def _get_data_range(self, usage):
        match usage:
            case 'val':
                self.start = ALL_DATA_NUM - self.N
                self.end = ALL_DATA_NUM
            case 'train':
                self.start = 0
                self.end = self.N
    
    def _loadU(self, ):
        U = []

        U1 = np.load(f'./TrainData/GridSize-{self.GridSize}/case-1/U.npy')[self.start:self.end].reshape(self.N, self.GridSize, self.GridSize)
        U2 = np.load(f'./TrainData/GridSize-{self.GridSize}/case-2/U.npy')[self.start:self.end].reshape(self.N, self.GridSize, self.GridSize)
        U3 = np.load(f'./TrainData/GridSize-{self.GridSize}/case-3/U.npy')[self.start:self.end].reshape(self.N, self.GridSize, self.GridSize)
        for i, case in enumerate(self.cases):
            if case == 1:
                U.append(U1[i])
            elif case == 2:
                U.append(U2[i])
            elif case == 3:
                U.append(U3[i])
        U = np.stack(U) 
        return torch.from_numpy(U)

    def _load_layouts(self):
        layouts = np.load(f'./TrainData/GridSize-{self.GridSize}/F.npy')[self.start:self.end]
        return torch.from_numpy(layouts)

    def _load_boundary(self):
        raise NotImplementedError

def c4_load_boundary(GridSize):
    return [
        torch.stack([
            torch.ones( GridSize, GridSize),
            torch.zeros(GridSize, GridSize),
            torch.zeros(GridSize, GridSize),
        ]),
        torch.stack([
            torch.zeros(GridSize, GridSize),
            torch.ones( GridSize, GridSize),
            torch.zeros(GridSize, GridSize),
        ]),
        torch.stack([
            torch.zeros(GridSize, GridSize),
            torch.zeros(GridSize, GridSize),
            torch.ones(GridSize, GridSize),
        ])
        ]
    

class C4TrainDs(AbstractClass):
    def __init__(self, *args, **kwargs):
        super(C4TrainDs, self).__init__(*args, **kwargs, usage = 'train')
        
    def _load_boundary(self, ):
        return [
        torch.stack([
            torch.ones( self.GridSize, self.GridSize) * 298,
            torch.zeros(self.GridSize, self.GridSize),
            torch.zeros(self.GridSize, self.GridSize),
        ]),
        torch.stack([
            torch.zeros(self.GridSize, self.GridSize),
            torch.ones(self.GridSize, self.GridSize) * 298,
            torch.zeros(self.GridSize, self.GridSize),
        ]),
        torch.stack([
            torch.zeros(self.GridSize, self.GridSize),
            torch.zeros(self.GridSize, self.GridSize),
            torch.ones(self.GridSize, self.GridSize) * 298,
        ])
        ]
    
    def __getitem__(self, index):
        data = self._assemble_data(index)
        layout = self.layouts[index][None, ...]
        return self._to(data), self._to(layout), self.cases[index]

class C4ValDs(C4TrainDs):
    def __init__(self, *args, **kwargs):
        super(C4TrainDs, self).__init__(*args, **kwargs, usage = 'val')
    
    def __getitem__(self, index):
        data = self._assemble_data(index)
        u = self.U[index][None, ...]
        return self._to(data), self._to(u)

class C2TrainDS(AbstractClass):     
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, usage='train')

    def _load_boundary(self):
        xx, yy = np.meshgrid(
            np.linspace(0, 0.1, self.GridSize),
            np.linspace(0, 0.1, self.GridSize)
        )
        return [
            torch.from_numpy(case1(xx, yy))[None, ...],
            torch.from_numpy(case2(xx, yy))[None, ...],
            torch.from_numpy(case3(xx, yy))[None, ...]
        ]
    
    def __getitem__(self, index):
        layout = self.layouts[index][None, ...]
        data = self._assemble_data(index)
        return self._to(data), self._to(layout), self.cases[index]

class C2ValDS(C2TrainDS):     
    def __init__(self, *args, **kwargs):
        super(C2TrainDS, self).__init__(*args, **kwargs, usage = 'val')

    def __getitem__(self, index):
        u = self.U[index][None, ...]
        data = self._assemble_data(index)
        return self._to(data), self._to(u)
        