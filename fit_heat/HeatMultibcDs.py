import torch
import numpy as np
from torch.utils.data import Dataset

def case1( x, y):
    mask = ((x == 0) + (y == 0) + (x == 0.1) + (y == 0.1)) > 0
    return mask * 298

def case2( x, y):
    mask = x == 0
    return mask * 298

def case3( x, y):
    mask = (y == 0) * (x >= 0.0495) * (x <= 0.0505)
    return mask * 298

class C5DS(Dataset):
    def __init__(self, area, GridSize, layouts, dtype, device, bd_cases=None):
        self.GridSize = GridSize
        self.area = area
        
        (self.left, self.bottom), (self.right, self.top) = area
        self.dx, self.dy = (self.right - self.left) / GridSize, (self.top - self.bottom) / GridSize

        self.layouts = layouts.to(dtype).to(device)

        xx, yy = np.meshgrid(
            np.linspace(self.left, self.right, self.GridSize),
            np.linspace(self.bottom, self.top, self.GridSize)
        )
        self.xx = torch.from_numpy(xx).to(dtype).to(device)
        self.yy = torch.from_numpy(yy).to(dtype).to(device)

        boundary_tensors = []
        boundaries = []
        if bd_cases is None:
            self.cases = np.random.choice([0, 1, 2], self.layouts.shape[0])
        else:
            self.cases = bd_cases
        for bd_case in self.cases:
            match bd_case:
                case 0:
                    boundary = case1(self.xx, self.yy)
                    boundary_tensor = torch.ones((4, self.GridSize, self.GridSize)) * 298

                case 1:
                    boundary = case2(self.xx, self.yy)
                    boundary_tensor = torch.stack([
                        torch.ones(self.GridSize, self.GridSize) * 298,
                        torch.zeros(self.GridSize, self.GridSize),
                        torch.zeros(self.GridSize, self.GridSize),
                        torch.zeros(self.GridSize, self.GridSize),
                    ])

                case 2:
                    boundary = case3(self.xx, self.yy)
                    boundary_tensor = torch.stack([
                        torch.zeros(self.GridSize, self.GridSize),
                        case3(torch.arange(self.left + self.dx/2, self.right, self.dx), 0).expand(self.GridSize, self.GridSize),
                        torch.zeros(self.GridSize, self.GridSize),
                        torch.zeros(self.GridSize, self.GridSize),
                    ])

            boundaries.append(boundary)
            boundary_tensors.append(boundary_tensor)

        self.boundaries = torch.stack(boundaries)
        self.boundary_tensors = torch.stack(boundary_tensors).to(dtype).to(device)
        
    def __len__(self):
        return self.layouts.shape[0]

    def __getitem__(self, index):
        layout = self.layouts[index]
        boundary = self.boundaries[index]
        boundary_tensor = self.boundary_tensors[index]
        data = torch.vstack([boundary_tensor, layout[None, ...]])
        
        return data, layout, boundary, self.cases[index]
    


class C2DS(Dataset):
    def __init__(self, area, GridSize, layouts, dtype, device, bd_cases=None):
        self.GridSize = GridSize
        self.area = area
        ((self.left, self.bottom), (self.right, self.top)) = area
        self.dx, self.dy = (self.right - self.left) / GridSize, (self.top - self.bottom) / GridSize

        self.layouts = layouts.to(dtype).to(device)

        xx, yy = np.meshgrid(
            np.linspace(self.left, self.right, self.GridSize),
            np.linspace(self.bottom, self.top, self.GridSize)
        )
        self.xx = torch.from_numpy(xx).to(dtype).to(device)
        self.yy = torch.from_numpy(yy).to(dtype).to(device)
        
        boundary_tensors = []
        boundaries = []
        if bd_cases is None:
            self.cases = np.random.choice([0, 1, 2], self.layouts.shape[0])
        else:
            self.cases = bd_cases
        for bd_case in self.cases:
            match bd_case:
                case 0:
                    boundary = case1(self.xx, self.yy)
                case 1:
                    boundary = case2(self.xx, self.yy)
                case 2:
                    boundary = case3(self.xx, self.yy)

            boundaries.append(boundary)
            boundary_tensors.append(torch.clone(boundary))

        self.boundaries = torch.stack(boundaries).to(dtype).to(device)
        self.boundary_tensors = torch.stack(boundary_tensors).to(dtype).to(device)
        
    def __len__(self):
        return self.layouts.shape[0]

    def __getitem__(self, index):
        layout = self.layouts[index]
        boundary = self.boundaries[index]
        boundary_tensor = self.boundary_tensors[index]

        # ###
        # normed_layout = (layout - layout.min()) / (layout.max() - layout.min())
        # ###
        
        data = torch.stack([boundary_tensor, layout])
        
        return data, layout, boundary, self.cases[index]