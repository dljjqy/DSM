import torch
import numpy as np
import random

from scipy.sparse import load_npz
from scipy.sparse.linalg import spsolve
from MyPlot import multi_heat_draw_img as draw_img
from utils import ChipLayout, layout2csv, coo2tensor
from BaseTrainer import BaseTrainer
from Generators import JacTorch

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, area, GridSize, layouts, dtype, device):
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
        self.cases = np.random.choice([0, 1, 2], self.layouts.shape[0])
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
    def __init__(self, area, GridSize, layouts, dtype, device):
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
        self.cases = np.random.choice([0, 1, 2], self.layouts.shape[0])
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
        data = torch.stack([boundary_tensor, layout])
        
        return data, layout, boundary, self.cases[index]

class Trainer(BaseTrainer):
    def __init__(
        self, method, maxiter, boundary_gap=0.001, chip_gap=0.001, *args, **kwargs
    ):
        self.method = method
        self.maxiter = maxiter
        self.boundary_gap = boundary_gap
        self.chip_gap = chip_gap
        super().__init__(*args, **kwargs)

        (left, bottom), (right, top) = self.area
        self.dx, self.dy = (right - left) / self.GridSize, (top - bottom) / self.GridSize

        self.data_start_index = random.randint(0, 10000- self.trainN - self.valN)
        self.init_generator()

    @property
    def name(self):
        return f"{self.tag}-{self.net.name}_{self.GridSize}_{sum(self.total_epochs)}_{self.batch_size}_{self.lr:.2e}_{self.method}-{self.maxiter}"
    
    def hyper_param_need2save(self):
        param = {
            'GridSize': self.GridSize,
            'method': self.method,
            'maxiter': self.maxiter,
            'area': self.area,
            'trainN': self.trainN,
            'valN': self.valN,
            'Batch_size': self.batch_size,
            'lr':self.lr,
            'epochs':self.total_epochs,
            'tag':self.tag,
            'net_kwargs': self.net_kwargs,
    
        }
        return param
    
    def init_generator(self):
        self.generators = []
        self.B = []
        self.A = []
        
        for bd_case in [0, 1, 2]:
            A_np = load_npz(f'./DLdata/heat/GridSize-{self.GridSize}/case-{bd_case}/A.npz')
            
            b = np.load(f'./DLdata/heat/GridSize-{self.GridSize}/case-{bd_case}/b.npy')

            A = coo2tensor(A_np.tocoo(), self.device, self.dtype)
            b = torch.from_numpy(b).to(self.dtype).to(self.device)

            self.A.append(A_np)
            self.B.append(b)

            self.generators.append(JacTorch(A[None, ...], self.device, self.dtype))
        
        self.B = torch.stack(self.B)

    def init_traindl(self):
        all_layouts = np.load(f'./DLdata/heat/GridSize-{self.GridSize}/F.npy')
        all_layouts = torch.from_numpy(all_layouts[self.data_start_index: self.data_start_index + self.trainN + self.valN])

        match self.net_kwargs['in_channels']:
            case 2:
                train_ds = C2DS(self.area, self.GridSize, all_layouts[:self.trainN], self.dtype, self.device)
            case 5:
                train_ds = C5DS(self.area, self.GridSize, all_layouts[:self.trainN], self.dtype, self.device)
         
        self.train_dl = DataLoader(
            train_ds, self.batch_size, shuffle=False, drop_last=False
        )
    
    def init_valdl(self):
        all_layouts = np.load(f'./DLdata/heat/GridSize-{self.GridSize}/F.npy')
        all_layouts = torch.from_numpy(all_layouts[self.data_start_index: self.data_start_index + self.trainN + self.valN])

        match self.net_kwargs['in_channels']:
            case 2:
                val_ds = C2DS(self.area, self.GridSize, all_layouts[self.trainN:], self.dtype, self.device)
            case 5:
                val_ds = C5DS(self.area, self.GridSize, all_layouts[self.trainN:], self.dtype, self.device)

        self.val_dl = DataLoader(
            val_ds, self.batch_size, shuffle=False, drop_last=False
        )

    def train_step(self, data, layouts, bd_cases, maxiter):
        pre = self.net(data)

        # Generate the label by Jac
        with torch.no_grad():
            label = []
            for i, bd_case in enumerate(bd_cases):
                bd_case = bd_case.item()
                b = layouts[i].reshape(-1) * self.dx * self.dy + self.B[bd_case]

                generator = self.generators[bd_case]
                label.append(generator(pre[i], b[None, ...], maxiter))
                
            label = torch.stack(label, dim=0).to(self.dtype).to(self.device)

        # Compute MSE
        loss = self.loss_fn(pre, label)
        return loss

    def val_step(self, data, layouts, boundaries, bd_cases, maxiter):
        jac_ans = []
        real_ans = []

        pre = self.net(data)

        for i, bd_case in enumerate(bd_cases):
            bd_case = bd_case.item()
            b = layouts[i].reshape(-1) * self.dx * self.dy + self.B[bd_case]

            generator = self.generators[bd_case]
            jac_ans.append(generator(pre[i], b[None, ...], maxiter))
            real_ans.append(
                spsolve(self.A[bd_case], b.cpu().numpy()).reshape(1, self.GridSize, self.GridSize)
            )
            
        jac_ans = torch.stack(jac_ans, dim=0).to(self.dtype).to(self.device)
        real_ans = torch.from_numpy(np.stack(real_ans)).to(self.dtype).to(self.device)

        val_jac_loss = self.loss_fn(jac_ans, pre)
        val_real_loss = self.loss_fn(real_ans, pre)

        return pre, layouts, boundaries, real_ans, val_jac_loss, val_real_loss

    def train_loop(self):
        self.net.train()
        loss_vals = []

        for data, layouts, boundaries, bd_cases in tqdm(self.train_dl, position=1, leave=False):
            loss_val = self.train_step(data, layouts, bd_cases, self.maxiter)

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            self.writer.add_scalar(
                "TrainSubiterLoss", loss_val.item(), self.train_global_idx
            )
            self.train_global_idx += 1

            loss_vals.append(loss_val.item())
        return np.array(loss_vals).mean()

    def val_loop(self):
        self.net.eval()
        jac_loss_vals = []
        real_loss_vals = []

        for data, layouts, boundaries, bd_cases in tqdm(self.val_dl, position=1, leave=False):
            batch_size = len(bd_cases)
            pre, layouts, boundaries, real_ans, jac_loss, real_loss = self.val_step(
                data, layouts, boundaries, bd_cases, self.maxiter
            )
            self.val_global_idx += 1

            jac_loss, real_loss = jac_loss.item(), real_loss.item()

            real_loss_vals.append(real_loss)
            jac_loss_vals.append(jac_loss)

            self.val_plot(batch_size, pre, real_ans, layouts, boundaries)
            self.writer.add_scalar("ValLoss", real_loss, self.val_global_idx)
            self.writer.add_scalar("ValJacLoss", jac_loss, self.val_global_idx)

        return np.array(real_loss_vals).mean(), np.array(jac_loss_vals).mean()

    def val_plot(self, batch_size, pre, ans, layouts, boundaries):
        pre = pre.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
        ans = ans.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
        layouts = layouts.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
        boundaries = boundaries.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
        k = random.choice(range(batch_size))
        fig = draw_img(
            f"ValFigure",
            layouts[k],
            boundaries[k],
            pre[k],
            ans[k],
            self.GridSize,
            0.1,
            None,
        )
        self.writer.add_figure(f"ValFigure", fig, self.val_global_idx)

if __name__ == "__main__":
    from SegModel import *
    GridSize = 128
    mission_name = "multibc_heat_o"

    trainer = Trainer(
        method="jac",
        maxiter=10,
        dtype=torch.float,
        device="cuda",

        area=((0, 0), (0.1, 0.1)),
        GridSize=GridSize,
        trainN=8000,
        valN=20,
        batch_size=5,
        net_kwargs=
        {
            'model_name': 'segmodel',
            "Block": 'ResBottleNeck',
            "planes": 4,
            "in_channels": 2,
            "classes": 1,
            "GridSize": GridSize,
            "layer_nums": [4, 4, 6, 6, 8],
            "adaptor_nums": [4, 4, 6, 6, 8],
            "factor": 2,
            "norm_method": "layer",
            "pool_method": "max",
            "padding": "same",
            "padding_mode": "replicate",
        },
        log_dir=f"./all_logs/{mission_name}",
        lr=1e-3,
        total_epochs=[200, 200],
        tag="2CFixedData",
        loss_fn=nn.functional.mse_loss,
        model_save_path=f"./model_save/{mission_name}",
        hyper_params_save_path=f"./hyper_parameters/{mission_name}",
        )
    
    trainer.fit_loop()

    # {
        #     'model_name': 'segmodel',
        #     "Block": 'ResBottleNeck',
        #     "planes": 10,
        #     "in_channels": 5,
        #     "classes": 1,
        #     "GridSize": GridSize,
        #     "layer_nums": [ 4, 6, 6, 8],
        #     "adaptor_nums": [ 4, 6, 6, 8],
        #     "factor": 2,
        #     "norm_method": "layer",
        #     "pool_method": "max",
        #     "padding": "same",
        #     "padding_mode": "replicate",
        # },
        
        # {
        #     'model_name':'unet++',
        #     "encoder_name":'resnet18',
        #     "encoder_weights":None,
        #     "in_channels":5,
        #     "classes":1,
        #     },
