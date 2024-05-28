import sys
sys.path.append('../')
import torch
import numpy as np
import random

from MyPlot import multi_heat_draw_img as draw_img
from utils import ChipLayout, layout2csv, coo2tensor
from BaseTrainer import BaseTrainer
from Generators import *
from HeatMultibcDs import *

from scipy.sparse.linalg import spsolve
from scipy.sparse import load_npz
from tqdm import tqdm
from torch.utils.data import DataLoader


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
        self.init_linearsys()

    @property
    def name(self):
        return f"{self.tag}-{self.net.name}-{self.GridSize}-{self.method}-{self.maxiter}-{self.trainN}-{self.batch_size}"

    def reboot(self):
        self.config_optimizer(self.lr)
        # self.init_generator_monitor()
        # self.init_traindl()
        # self.init_valdl()

    def epoch_reboot(self):
        # if self.global_epoch_idx % 50 == 0:
        self.init_traindl()
        self.init_valdl()
        pass

    def gen_mesh(self, area, GridSize):
        (self.left, self.bottom), (self.right, self.top) = area
        self.bd_x, self.bd_y = np.meshgrid(
            np.linspace(self.left, self.right, GridSize),
            np.linspace(self.bottom, self.top, GridSize),
        )
        self.dx, self.dy = (self.right - self.left) / GridSize, (self.top - self.bottom) / GridSize
        self.xx, self.yy = np.meshgrid(
            np.arange(self.left + self.dx / 2, self.right, self.dx),
            np.arange(self.bottom + self.dy / 2, self.top, self.dy),
        )
        self.dx = (self.right - self.left) / GridSize
        self.dy = (self.top - self.bottom) / GridSize
    
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

    def gen_layouts(self, DataN):
        df = layout2csv(
            DataN, self.area, self.GridSize, self.boundary_gap, self.chip_gap
        )
        layouts = []

        for _, data in df.groupby("idx"):
            info = data.values[:, 1:]
            chips_layout = ChipLayout(info)
            layout = chips_layout(self.xx, self.yy)
            layouts.append(torch.from_numpy(layout))
        layouts = torch.stack(layouts, dim=0)
        return layouts
    
    def init_linearsys(self):
        self.Anp = []
        self.Atorch = []
        self.B = []

        for bd_case in [0, 1, 2]:
            A_np = load_npz(f'./DLdata/GridSize-{self.GridSize}/case-{bd_case}/A.npz')
            self.Anp.append(A_np)
            
            A_torch = coo2tensor(A_np.tocoo(), self.device, self.dtype)
            self.Atorch.append(A_torch)

            b = np.load(f'./DLdata/GridSize-{self.GridSize}/case-{bd_case}/b.npy')
            b = torch.from_numpy(b).to(self.dtype).to(self.device)
            self.B.append(b)
        
        self.B = torch.stack(self.B)
    
    def init_monitor(self, bd_cases):
        with torch.no_grad():
            batched_A = []
            for c in bd_cases:
                c = c.item()
                batched_A.append(self.Atorch[c])
            batched_A = torch.stack(batched_A)

            monitor = BatchedMonitor(batched_A, self.dtype, self.device)
            return monitor

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

    def train_step(self, data, layouts, bd_cases, maxiter):
        batch_size = layouts.shape[0]
        
        # First, we need generator and B
        monitor = self.init_monitor(bd_cases)
        B = layouts.reshape(batch_size, -1) * self.dx * self.dy + self.B[bd_cases]
        B = B[..., None]
        
        # Prediction
        pre = self.net(data) + 298

        loss = monitor(pre, B)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.writer.add_scalar("Train-ResidualLoss", loss_val, self.train_global_idx)
        self.train_global_idx += 1

        return loss_val

    def val_step(self, data, layout, boundary, bd_cases, maxiter):
        batch_size = layout.shape[0]

        # Get generator and monitor
        monitor = self.init_monitor(bd_cases)
        B = layout.reshape(batch_size, -1) * self.dx * self.dy + self.B[bd_cases]
        B = B[..., None]

        # Prediction
        pre = self.net(data) + 298
        loss_val = monitor(pre, B).item()

        # # Get Real ans
        real_ans = []
        for i, bd_case in enumerate(bd_cases):
            b = B[i].cpu().numpy()  
            ans = spsolve(self.Anp[bd_case], b)
            real_ans.append(ans.reshape(1, self.GridSize, self.GridSize))

        real_ans = torch.from_numpy(np.stack(real_ans)).to(self.dtype).to(self.device)
        val_real_loss = self.loss_fn(real_ans, pre).item()

        self.writer.add_scalar("Val-Residual", loss_val, self.val_global_idx)
        self.writer.add_scalar("Val-RealLoss", val_real_loss, self.val_global_idx)
        self.val_plot(batch_size, pre, real_ans, layout, boundary)

        self.val_global_idx += 1

        return loss_val, val_real_loss

    def train_loop(self):
        self.net.train()
        
        errors = []
        for data, layouts, _, bd_cases in tqdm(self.train_dl, desc='Training Loop:', leave=False, position=1):
            error = self.train_step(data, layouts, bd_cases, self.maxiter)
            errors.append(error)
        
        self.lr_scheduler.step()
            
        return np.array(errors).mean()

    def val_loop(self):
        self.net.eval()

        errors, real_loss =  [], []       
        for data, layout, boundary, bd_cases in tqdm(self.val_dl, desc='Validation Loop', leave=False, position=2):

            error,  val_real_loss = self.val_step(
                data, layout, boundary, bd_cases, self.maxiter
            )

            errors.append(error)
            real_loss.append(val_real_loss)
            
        return np.array(real_loss).mean(), np.array(errors).mean()
    

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
    from torch.nn.functional import mse_loss
    GridSize = 128
    # mission_name = "heat_multibc"
    tag = "ResidualC2Plus298"

    trainer = Trainer(
        method="jac",
        maxiter=15,
        dtype=torch.float,
        device="cuda",
        area=((0, 0), (0.1, 0.1)),
        GridSize=GridSize,
        trainN=10000,
        valN=10,
        batch_size=5,
        net_kwargs=
        {
            'model_name': 'segmodel',
            "Block": 'ResBottleNeck',
            "planes": 6,
            "in_channels": 2,
            "classes": 1,
            "GridSize": GridSize,
            "layer_nums": [2, 2, 4, 6, 8],
            "adaptor_nums": [2, 2, 4, 6, 8],
            "factor": 2,
            "norm_method": "layer",
            "pool_method": "max",
            "padding": "same",
            "padding_mode": "reflect",
            "end_padding_mode": "reflect",
            "act":"tanh",
        },
        log_dir=f"./all_logs",
        lr=1e-3,
        total_epochs=[180],
        tag=tag,
        loss_fn=mse_loss,
        model_save_path=f"./model_save",
        hyper_params_save_path=f"./hyper_parameters",
        )
    
    trainer.fit_loop()