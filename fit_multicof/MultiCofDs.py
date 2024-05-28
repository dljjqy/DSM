import torch
import numpy as np
from scipy.sparse import load_npz
from utils import coo2tensor, coo2data
from torch.utils.data import Dataset

class C3Ds(Dataset):
    def __init__(self, start, DataN, area, GridSize, dtype, device, normliza=True):
        self.start = start
        self.area = area
        self.GridSize = GridSize
        (left, bottom), (right, top) = area
        dx = (right - left) / GridSize
        dy = (top - bottom) / GridSize
        self.xx, self.yy = np.meshgrid(
            np.arange(left + dx/2, right, dx),
            np.arange(bottom + dy/2, top, dy)
        )
        self.path = f'./DLdata/{GridSize}'
        self.DataN = DataN
        self.dtype = dtype
        self.device = device
        if normliza:
            self.a = -1 / 9.9            
            self.b = 10/9.9
        else:
            self.a = 1
            self.b = 0

    def __len__(self):
        return self.DataN
    
    def read_matrix(self, index):
        a = load_npz(f"{self.path}/a{self.start + index}.npz").tocoo()
        # i, v = coo2data(a)
        A = coo2tensor(a, self.device, self.dtype)
        return A

    def __getitem__(self, index):
        cof = np.load(f"{self.path}/c{self.start + index}.npy")
        # cof = self.a * cof + self.b
        u = np.load(f"{self.path}/u{self.start + index}.npy")
        b = np.load(f"{self.path}/b{self.start + index}.npy")

        A = self.read_matrix(index)
        b = torch.from_numpy(b).to(self.dtype).to(self.device)
        u = torch.from_numpy(u[np.newaxis, ...]).to(self.dtype).to(self.device)

        # normed_cof = (cof - 0.1) / 9.9
        data = np.stack([self.xx, self.yy, self.a * cof + self.b], axis=0)
        data = torch.from_numpy(data).to(self.dtype).to(self.device)
        return data, cof, A, b, u

class C1Ds(C3Ds):
    def __getitem__(self, index):
        cof = np.load(f"{self.path}/c{self.start + index}.npy")
        # cof = self.a * cof + self.b

        u = np.load(f"{self.path}/u{self.start + index}.npy")
        b = np.load(f"{self.path}/b{self.start + index}.npy")
        
        # i, v = self.read_matrix(index)
        A = self.read_matrix(index)
        
        b = torch.from_numpy(b).to(self.dtype).to(self.device)
        u = torch.from_numpy(u[np.newaxis, ...]).to(self.dtype).to(self.device)

        # normed_cof = (cof - 0.1) / 9.9

        data = torch.from_numpy(self.a * cof + self.b).to(self.dtype).to(self.device)
        return data[None, ...], cof, A, b, u