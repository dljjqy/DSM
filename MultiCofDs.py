import torch
import numpy as np
from scipy.sparse import load_npz
from utils import coo2tensor
from torch.utils.data import Dataset

class C3Ds(Dataset):
    def __init__(self, start, DataN, area, GridSize, dtype, device):
        self.start = start
        (left, bottom), (right, top) = area
        dx = (right - left) / GridSize
        dy = (top - bottom) / GridSize
        self.xx, self.yy = np.meshgrid(
            np.arange(left + dx/2, right, dx),
            np.arange(bottom + dy/2, top, dy)
        )
        self.path = f'./DLdata/allcofs/{GridSize}'
        self.DataN = DataN
        self.dtype = dtype
        self.device = device

    def __len__(self):
        return self.DataN
    
    def read_matrix(self, index):
        a = load_npz(f"{self.path}/a{self.start + index}.npz").tocoo()
        return coo2tensor(a, self.device, self.dtype)

    def __getitem__(self, index):
        cof = np.load(f"{self.path}/c{self.start + index}.npy")
        u = np.load(f"{self.path}/u{self.start + index}.npy")
        b = np.load(f"{self.path}/b{self.start + index}.npy")

        A = self.read_matrix(index)
        b = torch.from_numpy(b).to(self.dtype).to(self.device)
        u = torch.from_numpy(u[np.newaxis, ...]).to(self.dtype).to(self.device)

        # normed_cof = (cof - 0.1) / 9.9
        data = np.stack([self.xx, self.yy, cof], axis=0)
        data = torch.from_numpy(data).to(self.dtype).to(self.device)
        return data, cof, A, b, u

class C1Ds(C3Ds):
    def __getitem__(self, index):
        cof = np.load(f"{self.path}/c{self.start + index}.npy")
        u = np.load(f"{self.path}/u{self.start + index}.npy")
        b = np.load(f"{self.path}/b{self.start + index}.npy")
        
        A = self.read_matrix(index)
        b = torch.from_numpy(b).to(self.dtype).to(self.device)
        u = torch.from_numpy(u[np.newaxis, ...]).to(self.dtype).to(self.device)

        # normed_cof = (cof - 0.1) / 9.9

        data = torch.from_numpy(cof).to(self.dtype).to(self.device)
        return data[None, ...], cof, A, b, u