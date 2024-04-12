import torch
import numpy as np

from BaseTrainer import BaseTrainer
from MyPlot import multi_cof_draw_img as draw_img
from Generators import *
from MultiCofDs import *

from tqdm import tqdm
from torch.utils.data import DataLoader
    
class Trainer(BaseTrainer):
    def __init__(
        self,  
        method = 'jac',
        maxiter = 5,
        *args, **kwargs
        ):
        self.method = method
        self.maxiter = maxiter
        super().__init__(*args, **kwargs)
        
    @property
    def name(self):
        return f"{self.tag}-{self.GridSize}-{self.net.name}-{self.method}-{self.maxiter}-{self.trainN}-{self.batch_size}"
    
    def reboot(self):
        self.init_traindl()
        self.init_valdl()
        self.config_optimizer(self.lr)
        # self.method = 'cg'

    def epoch_reboot(self):
        pass
    
    def hyper_param_need2save(self):
        kwargs = {
            'GridSize': self.GridSize,
            'area': self.area,
            'trainN': self.trainN,
            'valN': self.valN,
            'Batch_size': self.batch_size,
            'lr':self.lr,
            'epochs':self.total_epochs,
            'method': self.method,
            'maxiter': self.maxiter,
            'name': self.name,
            'tag':self.tag,
            'net_kwargs': self.net_kwargs
        }
        return kwargs

    def init_traindl(self):
        self.start = np.random.randint(0, 30000 - self.trainN - self.valN)
        if self.net_kwargs['in_channels'] == 3:
            train_ds = C3Ds(self.start, self.trainN, self.area, self.GridSize, self.dtype, self.device) 
        elif self.net_kwargs['in_channels'] == 1:
            train_ds = C1Ds(self.start, self.trainN, self.area, self.GridSize, self.dtype, self.device) 
            
        self.train_dl = DataLoader(train_ds, self.batch_size)

    def init_valdl(self):         
        if self.net_kwargs['in_channels'] == 3:
            val_ds = C3Ds(self.start + self.trainN, self.valN, self.area, self.GridSize, self.dtype, self.device) 
        elif self.net_kwargs['in_channels'] == 1:
            val_ds = C1Ds(self.start + self.trainN, self.valN, self.area, self.GridSize, self.dtype, self.device) 
            
        self.val_dl = DataLoader(val_ds, self.batch_size)
    
    def init_generator_monitor(self, A):
        match self.method:
            case 'jac':
                generator = JacBatched(A, self.dtype, self.device)
            case 'cg':
                generator = CGBatched(A, self.dtype, self.device)
        monitor = BatchedMonitor(A, self.dtype, self.device)
        return generator, monitor

    def train_step(self, data, A, B, maxiter):
        # Get the generator and monitor
        generator, monitor = self.init_generator_monitor(A)
        
        # Do prediction
        pre = self.net(data)

        with torch.no_grad():
            labels = generator(
                        torch.clone(torch.detach(pre)), 
                        B[..., None], maxiter)
            error = monitor(pre, B[..., None]).item()

        loss = self.loss_fn(labels, pre)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.writer.add_scalar("Train-SubIterLoss", loss_val, self.train_global_idx)
        self.writer.add_scalar("Train-Error", error, self.train_global_idx)
        self.train_global_idx += 1

        return error, loss_val

    def val_step(self, data, A, B, U, maxiter):
        # Get the generator and monitor
        generator, monitor = self.init_generator_monitor(A)

        # Prediction
        pre = self.net(data)

        # Generate the labels
        labels = generator(pre, B[..., None], maxiter)

        val_subiter_loss = self.loss_fn(labels, pre).item()
        val_real_loss = self.loss_fn(U, pre).item()
        error = monitor(pre, B[..., None]).item()

        return pre, val_real_loss, val_subiter_loss, error

    def train_loop(self):
        self.net.train()
        errors = []
        for data, cofs, A, B, U in tqdm(self.train_dl, desc='Training Loop:', position=1, leave=False):
            error, loss_val = self.train_step(data, A, B, self.maxiter)

            errors.append(error)
        self.lr_scheduler.step()
        return np.array(errors).mean()
    
    def val_loop(self):
        self.net.eval()
        val_errors, real_loss_vals = [], []
        for data, cofs, A, B, U in tqdm(self.val_dl, desc='Validation Loop:', position=2, leave=False):
            pre, real_loss, subiter_loss, error = self.val_step(data, A, B, U, self.maxiter)

            val_errors.append(error)
            real_loss_vals.append(real_loss)

            self.writer.add_scalar("Val-RealLoss", real_loss, self.val_global_idx)
            self.writer.add_scalar("Val-SubIterLoss", subiter_loss, self.val_global_idx)
            self.writer.add_scalar("Val-Error", error, self.val_global_idx)
            self.val_plot(pre, U, cofs)
            self.val_global_idx += 1

        return np.array(real_loss_vals).mean(), np.array(val_errors).mean()

    def val_plot(self, pre, U, cofs):
        batch_size = U.shape[0]

        pre = pre.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
        sols = U.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
        cofs = cofs.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)

        k = np.random.choice(batch_size)
        fig = draw_img('Validation', pre[k], sols[k], cofs[k], self.GridSize, a=1)
        self.writer.add_figure(f"ValFigure", fig, self.val_global_idx)

if __name__ == '__main__':
    # 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    # GridSize = 192
    GridSize = 96
    mission_name = 'allcofs'
    tag = 'JJQC1'

    trainer = Trainer(
        method='jac',
        maxiter=15,
        area = ((0, 0), (1, 1)),
        GridSize=GridSize,
        trainN=10000,
        valN=50,
        batch_size=5,
        net_kwargs={
            'model_name': 'segmodel',
            'Block': "ResBottleNeck",
            'planes':6,
            'in_channels':1,
            'classes':1,
            'GridSize':GridSize,
            'layer_nums':   [4, 6, 6, 8, 8],
            'adaptor_nums': [4, 6, 6, 8, 8],
            'factor':2,
            'norm_method': 'layer',
            'pool_method':'max',
            'padding':'same',
            'padding_mode':'replicate',
        },
        log_dir=f'./all_logs/{mission_name}',
        lr=1e-3,
        loss_fn=torch.nn.functional.mse_loss,
        model_save_path=f'./model_save/{mission_name}',
        tag = tag,
        total_epochs=[150],
        device='cuda',
        dtype=torch.float,
        hyper_params_save_path=f'./hyper_parameters/{mission_name}'
    )
    trainer.fit_loop()

    

        


