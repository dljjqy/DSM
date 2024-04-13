import torch
import numpy as np
import torch.nn.functional as F
from utils import L2Loss, kappa, force, hard_encode
from MyPlot import multi_nonlinear_draw_img as draw_img
from Generators import PinnGenerator_Ju
from fit_nlinear.NLinearDs import *
from tqdm import tqdm
from random import uniform
from torch.utils.data import DataLoader
from BaseTrainer import BaseTrainer
    
class NConvTrainer(BaseTrainer):
    def __init__(
            self, 
            mu, 
            gd, 
            maxiter=5,
            picard_eps=1e-5,
            subitr_eps=1e-7,
            max_subitr_step=500,
            max_picard_step=30,
            *args, **kwargs):
        self.mu = mu
        self.gd = gd
        self.maxiter = maxiter
        self.picard_eps = picard_eps
        self.subitr_eps = subitr_eps
        self.max_subitr_step = max_subitr_step
        self.max_picard_step = max_picard_step

        super().__init__(*args, **kwargs)

        self.h = 1.0 / (self.GridSize - 1)
        self.l2_loss = L2Loss(self.h)
        self.picard_global_step = 0
        self.init_generator()
    
    def reboot(self):
        self.init_traindl()
        self.init_valdl()
        self.config_optimizer(self.lr)
    
    @property
    def name(self):
        return f'{self.tag}-GridSize:{self.GridSize}-mu:{self.mu}-maxiter:{self.maxiter}-trainN:{self.trainN}-bs:{self.batch_size}'
    
    def hyper_param_need2save(self):
        param = {
            'mu':self.mu,
            'gd':self.gd,
            'maxiter':self.maxiter,
            'picard_eps':self.picard_eps,
            'subitr_eps':self.subitr_eps,
            'max_subitr_step':self.max_subitr_step,
            'nax_picard_step':self.max_picard_step,
            'GridSize': self.GridSize,
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
            case 3:
                train_ds = C3FTDs(self.GridSize, self.dtype, self.device, self.trainN, self.area)
            case _:
                train_ds = C1FTDs(self.GridSize, self.dtype, self.device, self.trainN, self.area)
        self.train_dl = DataLoader(train_ds, self.batch_size, shuffle=True)
        
    def init_valdl(self):
        match self.net_kwargs['in_channels']:
            case 3:
                val_ds = C3FVDs(self.GridSize, self.dtype, self.device, self.area)
            case _:
                val_ds = C1FVDs(self.GridSize, self.dtype, self.device, self.area)
        self.val_dl = DataLoader(val_ds, 1, shuffle=True)
        
    def init_generator(self, w=None):
        generator = PinnGenerator_Ju(
            self.batch_size, self.GridSize, self.dtype, self.device, self.maxiter, self.area, w, self.mu, self.gd)
        return generator
    
    def train_step(self, x, f, generator):
        pre = self.net(x)
        with torch.no_grad():
            label = generator(
                torch.clone(torch.detach(pre)), 
                f)
        loss = self.loss_fn(pre, label)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.writer.add_scalar("Train-PinnLoss", loss_val, self.train_global_idx)
        self.train_global_idx += 1

        return pre, label, loss_val
    
    def picard_train_step(self, x, f, generator):
        for i in tqdm(range(self.max_subitr_step), position=3, leave=False, desc='Linear Sub-Iteration:'):
            pre, label, loss_val = self.train_step(x, f, generator)

            with torch.no_grad():
                w = torch.clone(torch.detach(hard_encode(pre, self.gd)))
                monitor = self.init_generator(w)

                y = monitor(pre, f)
                subiter_error = self.l2_loss(y, pre).item()

                # subiter_error = self.l2_loss(pre, label).item() 
                if subiter_error < self.subitr_eps:
                    break
        return pre
        
    def picard_loop(self, x, f, old_pre):
        for i in tqdm(range(self.max_picard_step), position=2, leave=False, desc='Picard Iteration:'):
            generator = self.init_generator(w=old_pre)

            new_pre = self.picard_train_step(x, f, generator)
            
            # picard_error = self.l2_loss(old_pre, new_pre).item()
            with torch.no_grad():
                w = torch.clone(torch.detach(hard_encode(new_pre, self.gd)))
                monitor = self.init_generator(w)
                label = monitor(new_pre, f)

                picard_error = self.l2_loss(hard_encode(label, self.gd), w).item()

                self.writer.add_scalar("Train-PicardError", picard_error, self.picard_global_step)
                self.picard_global_step += 1
           
                if picard_error <= self.picard_eps:
                    break
                else:
                    old_pre = torch.clone(torch.detach(new_pre))
                
        return picard_error

    def train_loop(self, ):
        self.net.train()
        train_picard_errors = []

        for x, f in tqdm(self.train_dl, position=1, leave=True, desc='Training Loop:'):
            with torch.no_grad():
                if self.picard_global_step == 0:
                    old_pre = torch.rand((self.batch_size, 1, self.GridSize, self.GridSize)).to(self.dtype).to(self.device) 
                else:
                    old_pre = torch.clone(torch.detach(hard_encode(self.net(x), self.gd)))

            picard_error = self.picard_loop(x, f, old_pre=old_pre)
            train_picard_errors.append(picard_error)

        return np.array(train_picard_errors).mean()

    def val_step(self, x, f, ans):
        pre = self.net(x)
        generator = self.init_generator(w = hard_encode(pre, self.gd))
        label = generator(pre, f) 
        
        val_loss = self.loss_fn(pre, label)
        real_loss = self.loss_fn(hard_encode(pre, self.gd), ans)

        self.writer.add_scalar("Val-Loss", val_loss.item(), self.val_global_idx)
        self.writer.add_scalar("Val-RealLoss", real_loss.item(), self.val_global_idx)

        return pre, val_loss.item(), real_loss.item()

    def val_loop(self):
        val_real_loss = []
        val_subitr_loss = []
        self.net.eval()
        with torch.no_grad():
            for x, f, ans in self.val_dl:
                pre, val_loss, real_loss = self.val_step(x, f, ans)
                
                val_subitr_loss.append(val_loss)
                val_real_loss.append(real_loss)
                self.val_global_idx += 1

            fig = self.val_plot(
                u = hard_encode(pre, self.gd),
                f = f,
                ans = ans,
                name = f"Val-{self.val_global_idx}")
            
            self.writer.add_figure("ValFigure", fig, self.val_global_idx)
            return np.array(val_subitr_loss).mean(), np.array(val_real_loss).mean()
        
    
    def val_plot(self, u, f, ans, name):
        u = u.reshape(self.GridSize, self.GridSize)
        u = u.detach().cpu().numpy()
        
        force = f.reshape(self.GridSize, self.GridSize)
        force = force.detach().cpu().numpy()

        ans = ans.reshape(self.GridSize, self.GridSize)
        ans = ans.detach().cpu().numpy()

        fig = draw_img(name, force, 0.1, u, ans, self.GridSize, 1, None)
        return fig
    
    

if __name__ == '__main__':
    mission_name = 'nlinear'
    tag = 'JuF_test1'
    trainer = NConvTrainer(
        gd=0, maxiter=5, mu=0.1,
        picard_eps=1e-7, subitr_eps=1e-8, 
        max_subitr_step=800, max_picard_step=100,
        dtype=torch.float, device='cuda',
        area=((0,0), (1,1)), GridSize=128,
        trainN=10000, valN=100, batch_size=5,
        net_kwargs={
            'model_name': 'varyunet',
            'in_channels':1,
            'classes':1,
            'features':8,
            'layers':5,
            'end_padding':'valid'
        },
        log_dir=f'./all_logs/{mission_name}',
        lr=1e-3, total_epochs=[150],
        tag=tag, loss_fn = F.mse_loss,
        model_save_path=f'./model_save/{mission_name}',
        hyper_params_save_path=f'./hyper_parameters/{mission_name}'
    )
    trainer.fit_loop()