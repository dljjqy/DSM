import numpy as np
import torch
import numpy as np
import random

from BaseTrainer import BaseTrainer
from utils import *
from MyPlot import *

from tqdm import tqdm

def u0(x, y):
    return np.sin(np.pi * x)*np.sin(np.pi * y)

def f0(x, y):
    return 2 * np.pi**2 * u0(x, y)

class ResidualLoss(torch.nn.Module):
    def __init__(self, dtype, device, h, loss_fn, gd=0):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.h = h
        self.loss_fn = loss_fn
        self.gd = gd

        self.k = self._get_kernel([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.hard_encode = lambda x: hard_encode(x, self.gd)

    def _get_kernel(self, k):
        k = torch.tensor(k, requires_grad=False)
        k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(self.dtype).to(self.device)
        return k
        
    def forward(self, pre, f):
        u = self.hard_encode(pre)
        y = F.conv2d(u, self.k)
        return self.loss_fn(y.squeeze(), f[..., 1:-1, 1:-1])

class JacLoss(torch.nn.Module):
    def __init__(self, dtype, device, h, loss_fn, gd=0):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.h = h
        self.loss_fn = loss_fn
        self.gd = gd

        self.k = self._get_kernel([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
        self.hard_encode = lambda x: hard_encode(x, self.gd)

    def _get_kernel(self, k):
        k = torch.tensor(k, requires_grad=False)
        k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(self.dtype).to(self.device)
        return k
        
    def forward(self, pre, f):
        with torch.no_grad():
            u = self.hard_encode(pre)
            w = F.conv2d(u, self.k) + 0.25 * self.h**2 * f[..., 1:-1, 1:-1]
        return self.loss_fn(pre, w)

class Trainer(BaseTrainer):
    def __init__(
        self, 
        method, 
        *args, **kwargs
    ):
        self.method = method
        super().__init__(*args, **kwargs)
        self.gen_mesh()

        # self.loss = ResidualLoss(self.dtype, self.device, self.h, self.loss_fn)
        match self.method:
            case 'jac':
                self.loss = JacLoss(self.dtype, self.device, self.h, self.loss_fn)
            case 'residual':
                self.loss = ResidualLoss(self.dtype, self.device, self.h, self.loss_fn)

        self.force = torch.from_numpy(f0(self.xx, self.yy)).to(self.dtype).to(self.device)
        self.ans = torch.from_numpy(u0(self.xx, self.yy)).to(self.dtype).to(self.device)

    @property
    def name(self):
        return f"{self.net.name}-{self.GridSize}-{self.method}-{self.trainN}"
    
    def l2(self, pre):
        diff = (pre.squeeze() - self.ans[1:-1, 1:-1]) ** 2 * self.h**2
        return torch.sqrt(diff.sum()).item()

    def gen_mesh(self):
        (self.left, self.bottom), (self.right, self.top) = self.area
        self.xx, self.yy = np.meshgrid(
            np.linspace(self.left, self.right, self.GridSize),
            np.linspace(self.bottom, self.top, self.GridSize),
        )
        self.h = (self.right - self.left) / (self.GridSize - 1)

    def reboot(self):
        self.config_optimizer(self.lr)

    def hyper_param_need2save(self):
        param = {
            'GridSize': self.GridSize,
            'method': self.method,
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
    
    def train_step(self):
        # Prediction
        force = torch.clone(torch.detach(self.force))
        data = torch.clone(torch.detach(self.force))[None, None, ...]
        # print(data.shape)

        pre = self.net(data)

        # Generate the label by Jac
        loss = self.loss(pre, force)
        error = self.l2(pre)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.writer.add_scalar("Train-ResidualLoss", loss_val, self.train_global_idx)
        self.writer.add_scalar("Train-Error", error, self.train_global_idx)
        self.train_global_idx += 1

        return loss_val

    def val_step(self, ):
        force = torch.clone(torch.detach(self.force))
        data = torch.clone(torch.detach(self.force))[None, None, ...]
        pre = self.net(data)
        
        # Prediction
        pre = self.net(data)
        loss_val = self.loss(pre, force).item()
        error = self.l2(pre)

        self.writer.add_scalar("Val-Residual", loss_val, self.val_global_idx)
        self.writer.add_scalar("Val-Error", error, self.val_global_idx)
        self.val_plot(pre)

        self.val_global_idx += 1

        return loss_val, error

    def train_loop(self):
        self.net.train()
        
        errors = []
        for _ in tqdm(range(self.trainN), desc='Training Loop:', leave=False, position=1):
            error = self.train_step()
            errors.append(error)
        
        self.lr_scheduler.step()
            
        return np.array(errors).mean()

    def val_loop(self):
        self.net.eval()

        errors, real_loss =  [], []       
        error,  val_real_loss = self.val_step()

        errors.append(error)
        real_loss.append(val_real_loss)
            
        return np.array(real_loss).mean(), np.array(errors).mean()

    def val_plot(self, pre):
        pre = hard_encode(pre, 0).cpu().numpy().reshape(self.GridSize, self.GridSize)
        ans = self.ans.cpu().numpy().squeeze()
        force = self.force.cpu().numpy().squeeze()

        fig = multi_cof_draw_img(f"ValFigure", pre, ans, force, self.GridSize, a=1,)
        self.writer.add_figure(f"ValFigure", fig, self.val_global_idx)

if __name__ == "__main__":
    from torch.nn.functional import mse_loss
    GridSize = 128
    mission_name = "demo"
    tag = "Residual"

    trainer = Trainer(
        method="jac",
        dtype=torch.float,
        device="cuda",
        area=((-1, -1), (1, 1)),
        GridSize=GridSize,
        trainN=100,
        valN=1,
        batch_size=1,
        net_kwargs=
        {
            'model_name': 'segmodel',
            "Block": 'ResBottleNeck',
            "planes": 6,
            "in_channels": 1,
            "classes": 1,
            "GridSize": GridSize,
            "layer_nums": [2, 2, 2, 2],
            "adaptor_nums": [2, 2, 2, 2],
            "factor": 2,
            "norm_method": "batch",
            "pool_method": "max",
            "padding": "same",
            "padding_mode": "reflect",
            "end_padding":"valid",
            "end_padding_mode": "reflect",
        },
        log_dir=f"./all_logs/{mission_name}",
        lr=1e-3,
        total_epochs=[150],
        tag=tag,
        loss_fn=mse_loss,
        model_save_path=f"./model_save/{mission_name}",
        hyper_params_save_path=f"./hyper_parameters/{mission_name}",
        )
    
    trainer.fit_loop()