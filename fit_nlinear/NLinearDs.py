import numpy as np
import torch
from torch.utils.data import  Dataset
from utils import force
from random import uniform
from itertools import product

class C1FTDs(Dataset):
    def __init__(self, GridSize, dtype, device, trainN, area = ((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize),
            np.linspace(bottom, top, GridSize)
        ) 
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)

        center_points = np.random.uniform(0.1, 0.9, (trainN, 2))
        self.center_points = torch.from_numpy(center_points)
        self.area = area
        self.trainN = trainN

    def __len__(self):
        return self.trainN

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)

        data = torch.clone(f[None, ...]).to(self.dtype).to(self.device)
        return data, f[None, ...]

class C1FVDs(Dataset):
    def __init__(self, GridSize, dtype, device, area = ((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize),
            np.linspace(bottom, top, GridSize)
        ) 
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)
        self.center_points = list(product(torch.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1)))
        self.area = area

        U = np.load(f'DLdata/{GridSize}/U.npy')
        self.U = torch.from_numpy(U).float()

    def __len__(self):
        return len(self.center_points)

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)

        data = torch.clone(f[None, ...]).to(self.dtype).to(self.device)
        ans = self.U[index].to(self.dtype).to(self.device)
        return data, f[None, ...], ans[None, ...]
    

class C3FTDs(Dataset):
    def __init__(self, GridSize, dtype, device, trainN, area = ((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize),
            np.linspace(bottom, top, GridSize)
        ) 
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)

        center_points = np.random.uniform(0.1, 0.9, (trainN, 2))
        self.center_points = torch.from_numpy(center_points)
        self.area = area
        self.trainN = trainN

    def __len__(self):
        return self.trainN

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)
        data = torch.stack([self.xx, self.yy, f], dim=0).to(self.dtype).to(self.device)
        return data, f[None, ...]

class C3FVDs(Dataset):
    def __init__(self, GridSize, dtype, device, area = ((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize),
            np.linspace(bottom, top, GridSize)
        ) 
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)
        self.center_points = list(product(torch.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1)))
        self.area = area

        U = np.load(f'DLdata/{GridSize}/U.npy')
        self.U = torch.from_numpy(U).float()

    def __len__(self):
        return len(self.center_points)

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)
        data = torch.stack([self.xx, self.yy, f], dim=0).to(self.dtype).to(self.device)
        ans = self.U[index].to(self.dtype).to(self.device)
        return data, f[None, ...], ans[None, ...]
    

class C2MuTDs(Dataset):
    def __init__(self, GridSize, dtype, device, trainN, area = ((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize),
            np.linspace(bottom, top, GridSize)
        ) 
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)

        center_points = np.random.uniform(0.1, 0.9, (trainN, 2))
        self.center_points = torch.from_numpy(center_points)
        self.area = area
        self.trainN = trainN

    def __len__(self):
        return self.trainN

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)
        mu = torch.ones_like(f) * uniform(0.1, 1)
        data = torch.stack([f, mu]).to(self.dtype).to(self.device)
        return data, f[None, ...], mu[None, ...]

class C2MuVDs(Dataset):
    def __init__(self, GridSize, dtype, device, area = ((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize),
            np.linspace(bottom, top, GridSize)
        ) 
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)
        self.center_points = list(product(torch.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1)))
        self.area = area

        U = np.load(f'DLdata/{GridSize}/U.npy')
        self.U = torch.from_numpy(U).float()

    def __len__(self):
        return len(self.center_points)

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)
        mu = torch.ones_like(f) * 0.1
        data = torch.stack([f, mu]).to(self.dtype).to(self.device)

        ans = self.U[index].to(self.dtype).to(self.device)
        return data, f[None, ...], mu[None, ...], ans[None, ...]
    

class C4MuTDs(Dataset):
    def __init__(self, GridSize, dtype, device, trainN, area = ((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize),
            np.linspace(bottom, top, GridSize)
        ) 
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)

        center_points = np.random.uniform(0.1, 0.9, (trainN, 2))
        self.center_points = torch.from_numpy(center_points)
        self.area = area
        self.trainN = trainN

    def __len__(self):
        return self.trainN

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)
        mu = torch.ones_like(f) * uniform(0.1, 1)
        data = torch.stack([self.xx, self.yy, f, mu]).to(self.dtype).to(self.device)
        return data, f[None, ...], mu[None, ...]

class C4MuVDs(Dataset):
    def __init__(self, GridSize, dtype, device, area = ((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize),
            np.linspace(bottom, top, GridSize)
        ) 
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)
        self.center_points = list(product(torch.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1)))
        self.area = area

        U = np.load(f'DLdata/{GridSize}/U.npy')
        self.U = torch.from_numpy(U).float()

    def __len__(self):
        return len(self.center_points)

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)
        mu = torch.ones_like(f) * 0.1
        data = torch.stack([self.xx, self.yy, f, mu]).to(self.dtype).to(self.device)

        ans = self.U[index].to(self.dtype).to(self.device)
        return data, f[None, ...], mu[None, ...], ans[None, ...]

