import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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
    def __init__(self, GridSize, N, dtype, device, bd_cases=None, usage='val'):
        super(AbstractClass, self).__init__()
        self.GridSize = GridSize
        self.N = N
        self.dtype = dtype
        self.device = device
        if bd_cases is None:
            self.cases = np.random.choice([1, 2, 3], N).astype(np.int32)
        else:
            self.cases = bd_cases

        self._get_data_range(usage)
        self.layouts = self._load_layouts()
        self.bd_tensors = self._load_boundary()
        self.B = self._loadB()
        match usage:
            case 'val':
                self.U = self._loadU()
            case 'train':
                pass
    
    def _to(self, x):
        return x.to(self.dtype).to(self.device)

    def __len__(self):
        return self.N
    
    def _assemble_data(self, index):
        layout = self.layouts[index]
        boundary_tensor = self.bd_tensors[self.cases[index]-1]
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
        U = np.zeros((self.N, self.GridSize, self.GridSize)) 

        U1 = np.load(f'./TrainData/GridSize-{self.GridSize}/case-1/U.npy')[self.start:self.end].reshape(self.N, self.GridSize, self.GridSize)
        U2 = np.load(f'./TrainData/GridSize-{self.GridSize}/case-2/U.npy')[self.start:self.end].reshape(self.N, self.GridSize, self.GridSize)
        U3 = np.load(f'./TrainData/GridSize-{self.GridSize}/case-3/U.npy')[self.start:self.end].reshape(self.N, self.GridSize, self.GridSize)
        U[np.bool_(self.cases == 1)] = U1[np.bool_(self.cases == 1)]
        U[np.bool_(self.cases == 2)] = U2[np.bool_(self.cases == 2)]
        U[np.bool_(self.cases == 3)] = U3[np.bool_(self.cases == 3)]
        return torch.from_numpy(U)

    def _load_layouts(self):
        layouts = np.load(f'./TrainData/GridSize-{self.GridSize}/F.npy')[self.start:self.end]
        return torch.from_numpy(layouts)

    def _loadB(self):
        B = np.zeros((self.N, self.GridSize, self.GridSize)) 
        B1 = np.load(f'./TrainData/GridSize-{self.GridSize}/case-{1}/B.npy')[self.start:self.end].reshape(self.N, self.GridSize, self.GridSize)
        B2 = np.load(f'./TrainData/GridSize-{self.GridSize}/case-{2}/B.npy')[self.start:self.end].reshape(self.N, self.GridSize, self.GridSize)
        B3 = np.load(f'./TrainData/GridSize-{self.GridSize}/case-{3}/B.npy')[self.start:self.end].reshape(self.N, self.GridSize, self.GridSize)

        B[np.bool_(self.cases == 1)] = B1[np.bool_(self.cases == 1)]
        B[np.bool_(self.cases == 2)] = B2[np.bool_(self.cases == 2)]
        B[np.bool_(self.cases == 3)] = B3[np.bool_(self.cases == 3)]
        return torch.from_numpy(B)  

    def _load_boundary(self):
        raise NotImplementedError

class C4TrainDs(AbstractClass):
    def __init__(self, *args, **kwargs):
        super(C4TrainDs, self).__init__(*args, **kwargs, usage = 'train')
        
    def _load_boundary(self, ):
        return [
        torch.stack([
            torch.ones( self.GridSize, self.GridSize),
            torch.zeros(self.GridSize, self.GridSize),
            torch.zeros(self.GridSize, self.GridSize),
        ]),
        torch.stack([
            torch.zeros(self.GridSize, self.GridSize),
            torch.ones(self.GridSize, self.GridSize),
            torch.zeros(self.GridSize, self.GridSize),
        ]),
        torch.stack([
            torch.zeros(self.GridSize, self.GridSize),
            torch.zeros(self.GridSize, self.GridSize),
            torch.ones(self.GridSize, self.GridSize),
        ])
        ]
    
    def __getitem__(self, index):
        b = self.B[index][None, ...]
        data = self._assemble_data(index)
        return self._to(data), self._to(b)

class C4ValDs(C4TrainDs):
    def __init__(self, *args, **kwargs):
        super(C4TrainDs, self).__init__(*args, **kwargs, usage = 'val')
    
    def __getitem__(self, index):
        data = self._assemble_data(index)
        u = self.U[index]
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
        b = self.B[index][None, ...]
        data = self._assemble_data(index)
        return self._to(data), self._to(b)

class C2ValDS(C2TrainDS):     
    def __init__(self, *args, **kwargs):
        super(C2TrainDS, self).__init__(*args, **kwargs, usage = 'val')

    def __getitem__(self, index):
        u = self.U[index]
        data = self._assemble_data(index)
        return self._to(data), self._to(u)

class C1TrainDS(AbstractClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, usage='train')
    
    def _load_boundary(self):
        pass

    def __getitem__(self, index):
        b = self.B[index][None, ...]
        data = torch.clone(b)
        return self._to(data), self._to(b)

class C1ValDS(C1TrainDS):
    def __init__(self, *args, **kwargs):
        super(C1TrainDS, self).__init__(*args, **kwargs, usage = 'val')

    def __getitem__(self, index):
        u = self.U[index]
        data = self.B[index][None, ...]
        return self._to(data), self._to(u)