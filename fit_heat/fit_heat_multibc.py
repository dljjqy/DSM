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
        self.init_generator_monitor()

    def epoch_reboot(self):
        self.init_traindl()
        self.init_valdl()

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
            A_np = load_npz(f'./DLdata/heat/GridSize-{self.GridSize}/case-{bd_case}/A.npz')
            self.Anp.append(A_np)
            
            A_torch = coo2tensor(A_np.tocoo(), self.device, self.dtype)
            self.Atorch.append(A_torch)

            b = np.load(f'./DLdata/heat/GridSize-{self.GridSize}/case-{bd_case}/b.npy')
            b = torch.from_numpy(b).to(self.dtype).to(self.device)
            self.B.append(b)
        
        self.B = torch.stack(self.B)
    
    def init_generator_monitor(self,):
        self.generator = []
        self.monitor = []

        for bd_case in [0, 1, 2]:
            self.generator.append(
                JacTorch(self.Atorch[bd_case], self.device, self.dtype)
            )    
            self.monitor.append(
                LinearMonitor(self.Atorch[bd_case], self.dtype, self.device)
            )

    # def init_generator_monitor(self, bd_cases):
    #     batched_A = []
    #     for c in bd_cases:
    #         c = c.item()
    #         batched_A.append(torch.clone(torch.detach(self.Atorch[c])))
    #     batched_A = torch.stack(batched_A)
    #     generator = JacBatched(batched_A, self.dtype, self.device)
    #     monitor = BatchedMonitor(batched_A, self.dtype, self.device)
    #     return generator, monitor

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
        generator, monitor = self.init_generator_monitor(bd_cases)
        B = layouts.reshape(batch_size, -1) * self.dx * self.dy + self.B[bd_cases]
        B = B[..., None]
        
        # Prediction
        pre = self.net(data)

        # Generate the label by Jac
        with torch.no_grad():
            label = generator(
                    torch.clone(torch.detach(pre)),
                    B, maxiter
                )
            error = monitor(pre, B).item()

        # with torch.no_grad():
        #     labels, errors = [], []
        #     for i in range(batch_size):
        #         labels.append(
        #             self.generator[i](pre[i], B[i], maxiter)
        #             )
        #         errors.append(
        #             self.monitor[i](pre[i], B[i]).item()
        #         )
        #     label = torch.stack(labels)
        #     error = np.array(errors).mean()
        # Compute MSE
        loss = self.loss_fn(pre, label)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.writer.add_scalar("Train-PinnLoss", loss_val, self.train_global_idx)
        self.writer.add_scalar("Train-Error", error, self.train_global_idx)
        self.train_global_idx += 1

        return error, loss_val

    def val_step(self, data, layout, boundary, bd_cases, maxiter):
        batch_size = layout.shape[0]

        # Get generator and monitor
        generator, monitor = self.init_generator_monitor(bd_cases)
        B = layout.reshape(batch_size, -1) * self.dx * self.dy + self.B[bd_cases]
        B = B[..., None]

        # Prediction
        pre = self.net(data)

        # Compute the label and error
        subiter_ans = generator(pre, B, maxiter)
        
        # with torch.no_grad():
        #     labels, errors = [], []
        #     for i in range(batch_size):
        #         labels.append(
        #             self.generator[i](pre[i], B[i], maxiter)
        #             )
        #         errors.append(
        #             self.monitor[i](pre[i], B[i]).item()
        #         )
        #     subiter_ans = torch.stack(labels)
        #     error = np.array(errors).mean()
        # # Get Real ans
        real_ans = []
        for i, bd_case in enumerate(bd_cases):
            b = B[i].cpu().numpy()  
            ans = spsolve(self.Anp[bd_case], b)
            real_ans.append(ans.reshape(1, self.GridSize, self.GridSize))

        real_ans = torch.from_numpy(np.stack(real_ans)).to(self.dtype).to(self.device)

        val_pinn_loss = self.loss_fn(subiter_ans, pre).item()
        val_real_loss = self.loss_fn(real_ans, pre).item()
        error = monitor(pre, B).item()

        self.writer.add_scalar("Val-Error", error, self.val_global_idx)
        self.writer.add_scalar("Val-RealLoss", val_real_loss, self.val_global_idx)
        self.writer.add_scalar("Val-PinnLoss", val_pinn_loss, self.val_global_idx)
        self.val_plot(batch_size, pre, real_ans, layout, boundary)

        self.val_global_idx += 1

        return error, val_pinn_loss, val_real_loss

    def train_loop(self):
        self.net.train()
        
        errors = []
        for data, layouts, _, bd_cases in tqdm(self.train_dl, desc='Training Loop:', leave=False, position=1):
            error, loss_val = self.train_step(data, layouts, bd_cases, self.maxiter)
            errors.append(error)
        
        self.lr_scheduler.step()
            
        return np.array(errors).mean()

    def val_loop(self):
        self.net.eval()

        errors, pinn_loss, real_loss = [], [], []       
        for data, layout, boundary, bd_cases in tqdm(self.val_dl, desc='Validation Loop', leave=False, position=2):

            error, val_pinn_loss, val_real_loss = self.val_step(
                data, layout, boundary, bd_cases, self.maxiter
            )

            errors.append(error)
            real_loss.append(val_real_loss)
            pinn_loss.append(val_pinn_loss)

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

    GridSize = 256
    mission_name = "heat_multibc"
    tag = "JJQC2ReSample"

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
            "layer_nums": [2, 2, 2, 2,],
            "adaptor_nums": [2, 2, 2, 2],
            "factor": 2,
            "norm_method": "layer",
            "pool_method": "max",
            "padding": "same",
            "padding_mode": "reflect",
            "end_padding_mode": "reflect",
        },
        log_dir=f"./all_logs/{mission_name}",
        lr=1e-3,
        total_epochs=[180],
        tag=tag,
        loss_fn=mse_loss,
        model_save_path=f"./model_save/{mission_name}",
        hyper_params_save_path=f"./hyper_parameters/{mission_name}",
        )
    
    trainer.fit_loop()