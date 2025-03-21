# Used to train CNN for solving the Poisson equation with parametric sources terms

import sys
sys.path.append('../')

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.lines as mlines

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
        *args, **kwargs
    ):
        self.method = method
        self.maxiter = maxiter
        super().__init__(*args, **kwargs)

        self.gen_mesh(self.area, self.GridSize)
        self.init_linearsys()

    @property
    def name(self):
        return f"{self.tag}"

    def reboot(self):
        self.config_optimizer(self.lr)

        self.init_traindl()
        self.init_valdl()

    def epoch_reboot(self):
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

    def init_traindl(self):
        match self.net_kwargs['in_channels']:
            case 1:
                train_ds = C1TrainDS(self.GridSize, self.trainN, self.dtype, self.device)
            case 2:
                train_ds = C2TrainDS(self.GridSize, self.trainN, self.dtype, self.device) 
            case 4:
                train_ds = C4TrainDs(self.GridSize, self.trainN, self.dtype, self.device) 

        self.train_dl = DataLoader(
            train_ds, self.batch_size, shuffle=False, drop_last=False
        )

    def init_valdl(self):
        layouts = self.gen_layouts(self.valN)
        match self.net_kwargs['in_channels']:
            case 1:
                val_ds = C1ValDS(self.GridSize, self.trainN, self.dtype, self.device)
            case 2:
                val_ds = C2ValDS(self.GridSize, self.trainN, self.dtype, self.device) 
            case 4:
                val_ds = C4ValDs(self.GridSize, self.trainN, self.dtype, self.device) 

        self.val_dl = DataLoader(
            val_ds, self.batch_size, shuffle=False, drop_last=False
        )

    def train_step(self, data, B, maxiter):
        pre = self.net(data)

        # Generate the label by Jac
        with torch.no_grad():
            label = self.generator(pre, B, maxiter)

        loss = self.loss_fn(pre, label)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.writer.add_scalar("Train-PinnLoss", loss_val, self.train_global_idx)
        self.train_global_idx += 1
        return loss_val

    def train_loop(self):
        self.net.train()
        loss_vals = []
        for data, b in tqdm(self.train_dl, desc='Training Loop:', leave=False, position=1):
            loss_val = self.train_step(data, b, self.maxiter)
            loss_vals.append(loss_val)
        self.lr_scheduler.step()
        return np.array(loss_vals).mean()

    def val_step(self, data, u):
        pre = self.net(data) 
        val_real_loss = self.loss_fn(pre, u).item()

        self.writer.add_scalar("Val-RealLoss", val_real_loss, self.val_global_idx)
        self.val_plot(pre, u)

        self.val_global_idx += 1
        return val_real_loss

    def val_loop(self):
        self.net.eval()
        real_loss = []       
        for data, u in tqdm(self.val_dl, desc='Validation Loop', leave=False, position=2):
            val_real_loss = self.val_step(data, u)
            real_loss.append(val_real_loss)
        return np.array(real_loss).mean(), np.array(real_loss).mean()
    

    def val_plot(self, pre, ans):
        pre = pre[0].cpu().numpy().reshape(self.GridSize, self.GridSize)
        ans = ans[0].cpu().numpy().reshape(self.GridSize, self.GridSize)
        fig = plt.figure()
        fig.suptitle('ValFigure', fontsize=20)
        fig.set_figheight(20)
        fig.set_figwidth(20)

        ax1 = fig.add_subplot(2, 2, 1, aspect="equal")
        ax2 = fig.add_subplot(2, 2, 2, aspect="equal")
        ax3 = fig.add_subplot(2, 2, 3, aspect="equal")
        ax4 = fig.add_subplot(2, 2, 4, aspect="equal")

        ax1.set_title(f'Prediction', fontsize=20)
        ctf = ax1.contourf(self.xx, self.yy, pre, alpha=1, cmap=cm.Spectral_r, levels=50)
        divider = make_axes_locatable(ax1)
        cax = divider.new_horizontal(size="2.5%", pad=0.1)
        fig.add_axes(cax)
        cbar = fig.colorbar(ctf, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=20)

        ax2.set_title(f'Reference', fontsize=20)
        ctf = ax2.contourf(self.xx, self.yy, ans, alpha=1, cmap=cm.Spectral_r, levels=50)
        divider = make_axes_locatable(ax2)
        cax = divider.new_horizontal(size="2.5%", pad=0.1)
        fig.add_axes(cax)
        cbar = fig.colorbar(ctf, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=20)

        ax3.set_title(f'Difference', fontsize=20)
        ctf = ax2.contourf(self.xx, self.yy, np.abs(ans - pre), alpha=1, cmap=cm.Spectral_r, levels=50)
        divider = make_axes_locatable(ax2)
        cax = divider.new_horizontal(size="2.5%", pad=0.1)
        fig.add_axes(cax)
        cbar = fig.colorbar(ctf, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=20)

        levels = np.linspace(ans.min(), ans.max(), 10)[2:-2] 
        ct1 = ax4.contour(self.xx, self.yy, pre, colors='r', linestyles='dashed', linewidths=1.5, levels=levels)
        ct2 = ax4.contour(self.xx, self.yy, ans, colors='b', linestyles='solid', linewidths=2, levels=levels)
        ax4.clabel(ct1, inline=False, fontsize=20)
        ax4.clabel(ct2, inline=False, fontsize=20)
        blue_line = mlines.Line2D([], [], color='blue', markersize=20, label='ref')
        red_line = mlines.Line2D([], [], color='red', markersize=20, label='pre')
        ax4.legend(handles=[blue_line, red_line], fontsize=20 )

        fig.tight_layout()
        self.writer.add_figure(f"ValFigure", fig, self.val_global_idx)
        plt.close(fig)


if __name__ == "__main__":
    from torch.nn.functional import mse_loss

    GridSize = 128
    tag = "UNetRelu-C2-float"

    trainer = Trainer(
        method="jac",
        maxiter=5,
        dtype=torch.float,
        device="cuda",
        area=((0, 0), (0.1, 0.1)),
        GridSize=GridSize,
        trainN=10000,
        valN=10,
        batch_size=5,
        net_kwargs={
            'model_name': 'UNet',
            'Block': "ResBottleNeck",
            'planes':6,
            'in_channels':2,
            'classes':1,
            'GridSize':GridSize,
            'layer_nums':   [2, 2, 4, 4, 6],
            'factor':2,
            'norm_method': 'layer',
            'pool_method':'max',
            'padding':'same',
            'padding_mode':'reflect',
            'end_padding':'same',
            'end_padding_mode':'reflect',
            'act':'relu'
        },
        log_dir=f"./all_logs",
        lr=1e-3,
        total_epochs=[150],
        tag=tag,
        loss_fn=mse_loss,
        model_save_path=f"./model_save",
        hyper_params_save_path=f"./hyper_parameters",
        )
    
    trainer.fit_loop()