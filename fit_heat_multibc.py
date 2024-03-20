import torch
import numpy as np
import random

from scipy.sparse.linalg import spsolve
from MyPlot import multi_heat_draw_img as draw_img
from UniformICD import UniformFVM
from Problems.BlockSourceProblem import BlockSourceProblem
from utils import ChipLayout, layout2csv, coo2tensor
from BaseTrainer import BaseTrainer
from Generators import JacTorch, CGTorch

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
        self, 
        method, 
        maxiter, 
        boundary_gap=0.001, 
        chip_gap=0.001, 
        *args, **kwargs
    ):
        self.method = method
        self.maxiter = maxiter
        self.boundary_gap = boundary_gap
        self.chip_gap = chip_gap
        super().__init__(*args, **kwargs)

        self.gen_mesh(self.area, self.GridSize)

    @property
    def name(self):
        return f"{self.tag}_{self.GridSize}_{sum(self.total_epochs)}_{self.trainN}_{self.batch_size}_{self.method}-{self.maxiter}"

    def reboot(self):
        self.init_traindl()
        self.init_valdl()
        self.config_optimizer()
        self.init_generator()

    def gen_mesh(self, area, GridSize):
        (self.left, self.bottom), (self.right, self.top) = area
        self.bd_x, self.bd_y = np.meshgrid(
            np.linspace(self.left, self.right, GridSize),
            np.linspace(self.bottom, self.top, GridSize),
        )
        self.dx = (self.right - self.left) / GridSize
        self.dy = (self.top - self.bottom) / GridSize

    def gen_layouts(self, DataN):
        df = layout2csv(
            DataN, self.area, self.GridSize, self.boundary_gap, self.chip_gap
        )
        xx, yy = np.meshgrid(
            np.arange(self.left + self.dx / 2, self.right, self.dx),
            np.arange(self.bottom + self.dy / 2, self.top, self.dy),
        )
        layouts = []

        for _, data in df.groupby("idx"):
            info = data.values[:, 1:]
            chips_layout = ChipLayout(info)
            layout = chips_layout(xx, yy)
            layouts.append(torch.from_numpy(layout))
        layouts = torch.stack(layouts, dim=0)
        return layouts
    
    def init_generator(self):
        print("initialize Generators...")
        self.generators = []
        self.B = []
        self.A = []

        for bd_case in [0, 1, 2]:
            problem = BlockSourceProblem(None, bd_case, 298, self.GridSize, self.area)
            solver = UniformFVM(self.area, self.GridSize, self.GridSize, problem)
            mat, b = solver.get_A()
            a = coo2tensor(mat.tocoo(), self.device, self.dtype)
            b = torch.from_numpy(b).to(self.dtype).to(self.device)

            self.B.append(b)
            self.A.append(mat)

            match self.method:
                case 'jac':
                    self.generators.append(JacTorch(a, self.device, self.dtype) )
                case 'cg':
                    self.generators.append(CGTorch(a, self.device, self.dtype) )

        self.B = torch.stack(self.B)
        print("We have Generators now, Start training...")

    def init_traindl(self):
        layouts = self.gen_layouts(self.trainN)
        match self.net_kwargs['in_channels']:
            case 2:
                train_ds = C2DS(self.area, self.GridSize, layouts, self.dtype, self.device)
            case 5:
                train_ds = C5DS(self.area, self.GridSize, layouts, self.dtype, self.device)

        self.train_dl = DataLoader(
            train_ds, self.batch_size, shuffle=False, drop_last=False
        )

    def init_valdl(self):
        layouts = self.gen_layouts(self.valN)
        match self.net_kwargs['in_channels']:
            case 2:
                val_ds = C2DS(self.area, self.GridSize, layouts, self.dtype, self.device)
            case 5:
                val_ds = C5DS(self.area, self.GridSize, layouts, self.dtype, self.device)

        self.val_dl = DataLoader(
            val_ds, self.batch_size, shuffle=False, drop_last=False
        )

    def train_step(self, data, layout, boundary, bd_cases, maxiter):
        # Prediction
        pre = self.net(data)

        # Generate the label by Jac
        with torch.no_grad():
            label = []
            for i, bd_case in enumerate(bd_cases):
                generator = self.generators[bd_case]
                b = layout[i].reshape(-1) * self.dx * self.dy + self.B[bd_case]

                label.append(
                    generator(
                        torch.clone(torch.detach(pre[i])), 
                        torch.clone(torch.detach(b[None, ...])), 
                        maxiter
                    )
                )

            label = torch.stack(label, dim=0).to(self.dtype).to(self.device)

        # Compute MSE
        loss = self.loss_fn(pre, label)
        return loss

    def val_step(self, data, layout, boundary, bd_cases, maxiter):
        # Prediction
        pre = self.net(data)

        jac_ans, real_ans = [], []
        for i, bd_case in enumerate(bd_cases):
            b = layout[i].reshape(-1) * self.dx * self.dy + self.B[bd_case]
            
            generator = self.generators[bd_case]
            jac_ans.append(generator(pre[i], b[None, ...], maxiter))

            real_ans.append(
                spsolve(
                    self.A[bd_case], 
                    b.cpu().numpy()
                ).reshape(1, self.GridSize, self.GridSize)
            )

        jac_ans = torch.stack(jac_ans, dim=0)
        real_ans = torch.from_numpy(np.stack(real_ans)).to(self.dtype).to(self.device)

        val_jac_loss = self.loss_fn(jac_ans, pre).item()
        val_real_loss = self.loss_fn(real_ans, pre).item()

        return pre, boundary, real_ans, val_jac_loss, val_real_loss

    def train_loop(self):
        self.net.train()
        loss_vals = []
        for data, layout, boundary, bd_cases in tqdm(self.train_dl, desc='Training Loop:', leave=False, position=1):
            loss = self.train_step(data, layout, boundary, bd_cases, self.maxiter)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_val = loss.item()
            
            self.writer.add_scalar(
                "TrainJacLoss", loss_val, self.train_global_idx
            )
            self.train_global_idx += 1

            loss_vals.append(loss_val)
            
        return np.array(loss_vals).mean()

    def val_loop(self):

        self.net.eval()
        jac_loss, real_loss = [], []        
        for data, layout, boundary, bd_cases in tqdm(self.val_dl, desc='Validation Loop', leave=False, position=2):
            batch_size = layout.shape[0]

            pre, boundary, real_ans, jac_loss, real_loss = self.val_step(
                data, layout, boundary, bd_cases, self.maxiter
            )

            jac_loss, real_loss = jac_loss.item(), real_loss.item()

            real_loss.append(real_loss)
            jac_loss.append(jac_loss)

            self.writer.add_scalar("ValLoss", real_loss, self.val_global_idx)
            self.writer.add_scalar("ValJacLoss", jac_loss, self.val_global_idx)

            pre = pre.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
            ans = real_ans.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
            layouts = layouts.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
            
            k = random.choice(range(batch_size))
            fig = draw_img(
                f"ValFigure",
                layouts[k],
                boundary[k],
                pre[k],
                ans[k],
                self.GridSize,
                0.1,
            )
            self.writer.add_figure(
                f"ValFigure-{self.val_global_idx}", 
                fig, 
                self.val_global_idx
                )
            self.val_global_idx += 1

        return np.array(real_loss).mean(), np.array(jac_loss).mean()


if __name__ == "__main__":
    from torch.nn.functional import mse_loss

    GridSize = 128
    mission_name = "multibc_layout_o"
    tag = "C2Test"

    trainer = Trainer(
        method="jac",
        maxiter=5,
        dtype=torch.float,
        device="cuda",
        area=((0, 0), (0.1, 0.1)),
        GridSize=GridSize,
        trainN=5000,
        valN=10,
        batch_size=5,
        net_kwargs=
        {
            'model_name': 'segmodel',
            "Block": 'ResBasic',
            "planes": 8,
            "in_channels": 2,
            "classes": 1,
            "GridSize": GridSize,
            "layer_nums": [2, 2, 4, 6, 8],
            "adaptor_nums": [2, 2, 4, 6, 8],
            "factor": 2,
            "norm_method": "layer",
            "pool_method": "max",
            "padding": "same",
            "padding_mode": "replicate",
        },
        log_dir=f"./all_logs/{mission_name}",
        lr=1e-3,
        total_epochs=[150, 150],
        tag=tag,
        loss_fn=mse_loss,
        model_save_path=f"./model_save/{mission_name}",
        hyper_params_save_path=f"./hyper_parameters/{mission_name}",
        )
    
    trainer.fit_loop()