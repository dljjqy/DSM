import torch
import numpy as np
import torch.nn.functional as F
from utils import L2Loss, kappa, force, hard_encode
from MyPlot import my_plot
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
                train_ds = C3TrainDs(self.GridSize, self.dtype, self.device, self.trainN, self.area)
            case _:
                train_ds = C1TrainDs(self.GridSize, self.dtype, self.device, self.trainN, self.area)
        self.train_dl = DataLoader(train_ds, self.batch_size, shuffle=True)
        
    def init_valdl(self):
        match self.net_kwargs['in_channels']:
            case 3:
                val_ds = C3ValDs(self.GridSize, self.dtype, self.device, self.area)
            case _:
                val_ds = C1ValDs(self.GridSize, self.dtype, self.device, self.area)
        self.val_dl = DataLoader(val_ds, 1, shuffle=True)
        
    def init_generator(self, w=None):
        self.generator = PinnGenerator(
            self.batch_size, self.GridSize, self.dtype, self.device, self.maxiter, self.area, w, self.mu, self.gd)
    
    def val_plot(self, u, f, ans, name):
        u = u.reshape(self.GridSize, self.GridSize)
        u = u.detach().cpu().numpy()
        
        force = f.reshape(self.GridSize, self.GridSize)
        force = force.detach().cpu().numpy()

        ans = ans.reshape(self.GridSize, self.GridSize)
        ans = ans.detach().cpu().numpy()

        fig = my_plot(self.GridSize, u, ans, force, name, 1.0)
        return fig
    
    def train_step(self, x, f):
        pre = self.net(x)
        with torch.no_grad():
            label = self.generator(
                torch.clone(torch.detach(pre)), 
                f)
        train_loss = self.loss_fn(pre, label)
        
        self.optimizer.zero_grad()

        train_loss.backward()
        self.optimizer.step()
        return pre, label, train_loss

    def train_loop(self, ):
        self.net.train()
        train_picard_loss = []

        for x, f in tqdm(self.train_dl, position=1, leave=True, desc='Training Loop:'):
            with torch.no_grad():
                if self.global_epoch_idx == 0:
                    old_pre = torch.ones((self.batch_size, 1, self.GridSize, self.GridSize)).to(self.dtype).to(self.device) 
                else:
                    old_pre = torch.clone(torch.detach(hard_encode(self.net(x), self.gd)))

            picard_loss = self.picard_loop(x, f, old_pre=old_pre)
            train_picard_loss.append(picard_loss)

        return np.array(train_picard_loss).mean()

    def val_step(self, x, f, ans):
        pre = self.net(x)
        self.init_generator(w = hard_encode(pre, self.gd))
        label = self.generator(pre, f) 
        
        val_loss = self.loss_fn(pre, label)
        real_loss = self.loss_fn(hard_encode(pre, self.gd), ans)

        self.writer.add_scalar("ValLoss", val_loss.item(), self.val_global_idx)
        self.writer.add_scalar("ValRealLoss", real_loss.item(), self.val_global_idx)

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
        
    def picard_train_step(self, x, f):
        for i in tqdm(range(self.max_subitr_step), position=3, leave=False, desc='Linear Sub-Iteration:'):
            pre, label, train_loss = self.train_step(x, f)

            self.train_global_idx += 1
            self.writer.add_scalar("TrainSubitrLoss", train_loss.item(), self.train_global_idx)

            subitr_error = self.l2_loss(pre, label).item() 
            if subitr_error < self.subitr_eps:
                break
        return hard_encode(pre, self.gd)
        
    def picard_loop(self, x, f, old_pre):
        self.init_generator(w=old_pre)
        for i in tqdm(range(self.max_picard_step), position=2, leave=False, desc='Picard Iteration:'):
            new_pre = self.picard_train_step(x, f)
            
            picard_error = self.l2_loss(old_pre, new_pre).item()
            self.writer.add_scalar("TrainPicardLoss", picard_error, self.picard_global_step)
            self.picard_global_step += 1
           
            old_pre = new_pre
            if picard_error <= self.picard_eps:
                break
        return picard_error
    

if __name__ == '__main__':
    trainer = NConvTrainer(
        mu=0.1, gd=0, maxiter=5,
        picard_eps=1e-7, subitr_eps=1e-8, 
        max_subitr_step=800, max_picard_step=50,
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
        log_dir='./all_logs/test',
        lr=1e-3, total_epochs=[150],
        tag='more', loss_fn = F.mse_loss,
        model_save_path='./model_save/test',
        hyper_params_save_path='./hyper_parameters/nlinear/test'
    )
    trainer.fit_loop()