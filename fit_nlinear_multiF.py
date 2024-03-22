import torch
import numpy as np
from MyPlot import multi_nonlinear_draw_img as draw_img
from tqdm import tqdm
from Generators import PinnGenerator
from torch.utils.data import DataLoader
import torch.nn.functional as F

from BaseTrainer import BaseTrainer
from NLinearDs import C1FTDs, C1FVDs, C3FTDs, C3FVDs
from utils import hard_encode, L2Loss
from random import choice
from SegModel import *

class NConvTrainer(BaseTrainer):
    def __init__(self,  
            gd=0, mu=0.1, maxiter=5,
            picard_eps = 1e-7,
            max_picard_step = 20,
            *args, **kwargs,
            ):
        self.gd = gd
        self.mu = mu
        self.maxiter = maxiter
        self.picard_eps = picard_eps
        self.max_picard_step = max_picard_step
        super().__init__(*args, **kwargs)
        self.h = 1.0 / (self.GridSize - 1)
        
        self.picard_global_idx = 0
        self.prev_net = None
        self.monitor = L2Loss(self.h)

    def init_traindl(self):
        match self.net_kwargs['in_channels']:
            case 1:
                train_ds = C1FTDs(self.GridSize, self.dtype, self.device, self.trainN, self.area)
            case 3:
                train_ds = C3FTDs(self.GridSize, self.dtype, self.device, self.trainN, self.area)
        train_dl = DataLoader(train_ds, self.batch_size, shuffle=True)
        return train_dl
    
    def init_valdl(self):
        match self.net_kwargs['in_channels']:
            case 1:
                val_ds = C1FVDs(self.GridSize, self.dtype, self.device, self.area)
            case 3:
                val_ds = C3FVDs(self.GridSize, self.dtype, self.device, self.area)
        
        val_dl = DataLoader(val_ds, self.batch_size, shuffle=True)
        return val_dl
    
    def init_generator(self, prev_net):
        self.generator = PinnGenerator(self.GridSize, self.device, self.maxiter, self.area, prev_net, self.gd)
    
    def train_step(self, x, f):
        pre = self.net(x)
        with torch.no_grad():
            label = self.generator(x.detach(), pre, f.detach(), self.mu)
        loss = self.loss_fn(pre, label)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.writer.add_scalar("Train-SubIterLoss", loss_val, self.train_global_idx)
        self.train_global_idx += 1
        
        return loss_val

    def train_loop(self, train_dl):
        self.net.train()
        loss_vals = []
        for x, f in tqdm(train_dl, position=1, leave=False):
            self.optimizer.zero_grad()
            loss_val = self.train_step(x, f)
            
            loss_vals.append(loss_val)
        self.lr_scheduler.step()

        return np.array(loss_vals).mean()

    def val_step(self, x, f, ans):
        pre = self.net(x)

        label = self.generator(x, pre, f, self.mu) 
        val_loss = self.loss_fn(pre, label)
        real_loss = self.loss_fn(hard_encode(pre, self.gd), ans)
        # error = self.monitor(old_pre, pre).item()

        self.writer.add_scalar("Val-SubIterLoss", val_loss.item(), self.val_global_idx)
        self.writer.add_scalar("Val-Error", val_loss.item(), self.val_global_idx)
        self.writer.add_scalar("Val-RealLoss", real_loss.item(), self.val_global_idx)
        self.val_plot(u = hard_encode(pre, self.gd), f = f,ans = ans,name = f"Val-{self.val_global_idx}")
        
        self.val_global_idx += 1
        return val_loss.item(), real_loss.item()

    def val_loop(self, val_dl):
        self.net.eval()

        val_loss_vals, real_loss_vals, errors = [], [], []
        for x, f, ans in val_dl:
            loss_val, real_loss_val = self.val_step(x, f, ans)

            val_loss_vals.append(loss_val)      
            real_loss_vals.append(real_loss_val)
            # errors.append(error)

        return np.array(val_loss_vals).mean(), np.array(real_loss_vals).mean()

    def val_plot(self, u, f, ans, name):
        batch_size = u.shape[0]
        u = u.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
        f = f.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
        ans = ans.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)

        k = choice(range(batch_size))
        fig = draw_img(name, f[k], self.mu, u[k], ans[k], self.GridSize, a=1)
        self.writer.add_figure("ValFigure", fig, self.val_global_idx)

    def picard_step(self, epoch, prev_net=None):
        train_dl = self.init_traindl()
        val_dl = self.init_valdl()
        self.init_network(self.net_kwargs)
        self.config_optimizer(self.lr)
        self.init_generator(prev_net)
        
        for _ in tqdm(range(epoch), position=0, leave=True, 
                    desc=f"Picard Step: {self.picard_global_idx}"):
            self.net.train()
            new_train_loss = self.train_loop(train_dl)
            self.save_best_train_error(new_train_loss)


            with torch.no_grad():
                val_loss, val_real_loss = self.val_loop(val_dl)
                # self.save_best_val_error(val_error)
                self.save_best_val_real(val_real_loss)

        self.picard_global_idx += 1
        torch.save(self.net, f"{self.model_save_path}/last_picard.pt")

        return self.net
                
    def fit_loop(self):
        prev_net = self.picard_step(self.total_epochs[0], None)
        for i in range(1, len(self.total_epochs)):
            prev_net = self.picard_step(self.total_epochs[i], prev_net)


if __name__ == '__main__':
    mission_name = 'Nonlinear-F'
    tag = 'test'

    trainer = NConvTrainer(
        mu=0.1, gd=0, maxiter=5, 
        picard_eps=1e-8,
        max_picard_step=30,
        dtype=torch.float, device='cuda',
        area=((0,0), (1,1)), GridSize=128,
        trainN=2000, valN=100, batch_size=10,
        net_kwargs={
            'model_name': 'varyunet',
            'in_channels':1,
            'classes':1,
            'features':8,
            'layers':5,
            'end_padding':'valid'
        },
        log_dir=f'./all_logs/{mission_name}',
        lr=1e-3, total_epochs=[5, 5, 5, 5],
        tag=tag, loss_fn = F.mse_loss,
        model_save_path=f'./model_save/{mission_name}',
        hyper_params_save_path=f'./hyper_parameters/{mission_name}'
    )
    trainer.fit_loop()