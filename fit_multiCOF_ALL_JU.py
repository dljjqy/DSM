import torch
import numpy as np

from utils import coo2tensor, L2Loss
from MyPlot import multi_cof_draw_img as draw_img
from Generators import JacBatched, CGBatched
from BaseTrainer import BaseTrainer
from scipy.sparse import load_npz

from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

class C3Ds(Dataset):
    def __init__(self, start, DataN, area, GridSize, dtype, device):
        self.start = start
        (left, bottom), (right, top) = area
        dx = (right - left) / GridSize
        dy = (top - bottom) / GridSize
        self.xx, self.yy = np.meshgrid(
            np.arange(left + dx/2, right, dx),
            np.arange(bottom + dy/2, top, dy)
        )
        self.path = f'./DLdata/allcofs/{GridSize}'
        self.DataN = DataN
        self.dtype = dtype
        self.device = device

    def __len__(self):
        return self.DataN
    
    def read_matrix(self, index):
        a = load_npz(f"{self.path}/a{self.start + index}.npz").tocoo()
        return coo2tensor(a, self.device, self.dtype)

    def __getitem__(self, index):
        cof = np.load(f"{self.path}/c{self.start + index}.npy")
        u = np.load(f"{self.path}/u{self.start + index}.npy")
        b = np.load(f"{self.path}/b{self.start + index}.npy")

        A = self.read_matrix(index)
        b = torch.from_numpy(b).to(self.dtype).to(self.device)
        u = torch.from_numpy(u[np.newaxis, ...]).to(self.dtype).to(self.device)

        data = np.stack([self.xx, self.yy, cof], axis=0)
        data = torch.from_numpy(data).to(self.dtype).to(self.device)
        return data, cof, A, b, u

class C1Ds(C3Ds):
    def __getitem__(self, index):
        cof = np.load(f"{self.path}/c{self.start + index}.npy")
        u = np.load(f"{self.path}/u{self.start + index}.npy")
        b = np.load(f"{self.path}/b{self.start + index}.npy")
        
        A = self.read_matrix(index)
        b = torch.from_numpy(b).to(self.dtype).to(self.device)
        u = torch.from_numpy(u[np.newaxis, ...]).to(self.dtype).to(self.device)
        data = torch.from_numpy(cof).to(self.dtype).to(self.device)
        return data, cof, A, b, u
    
class Trainer(BaseTrainer):
    def __init__(
        self,  
        method = 'jac',
        maxiter = 5,
        eps=1e-8,
        max_subiter_steps=800,
        *args, **kwargs
        ):
        self.method = method
        self.eps = eps
        self.max_subiter_steps = max_subiter_steps
        self.maxiter = maxiter
        super().__init__(*args, **kwargs)

        self.l2 = L2Loss(h = 1 / self.GridSize)
        self.global_subiter_step = 0
        
    @property
    def name(self):
        return f"{self.tag}-{self.GridSize}-{self.net.name}-{self.method}-{self.maxiter}-{self.trainN}-{self.batch_size}"
    
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
        self.start = np.random.randint(0, 50000 - self.trainN - self.valN)
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
    
    def init_generator(self, A):
        match self.method:
            case 'jac':
                generator = JacBatched(A, self.dtype, self.device)
            case 'cg':
                generator = CGBatched(A, self.dtype, self.device)
        return generator

    def train_step(self, data, A, B, maxiter):
        generator = self.init_generator(A)
        old_pre = self.net(data)
        for _ in tqdm(range(self.max_subiter_steps), desc='One Step Loop: ', position=2, leave=False):
            with torch.no_grad():
                jac_ans = generator(
                        torch.clone(torch.detach(old_pre)), 
                        B[..., None], maxiter
                        )
            loss = self.loss_fn(jac_ans, old_pre)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()
            new_pre = self.net(data)
            error = self.l2(new_pre, old_pre).item()

            self.writer.add_scalar("TrainSubIterLoss", loss_val, self.global_subiter_step)
            self.writer.add_scalar("TrainSubIterError", error, self.global_subiter_step)
            self.global_subiter_step += 1

            if error < self.eps:
                break
            else:
                old_pre = new_pre

        return error, loss_val

    def val_step(self, data, A, B, U, maxiter):
        with torch.no_grad():
            pre = self.net(data)

            val_real_loss = self.loss_fn(U, pre)
            
            generator = self.init_generator(A)
            jac_ans = generator(pre, B[..., None], maxiter)

            val_jac_loss = self.loss_fn(jac_ans, pre)

        return pre, val_real_loss.item(), val_jac_loss.item() 

    def train_loop(self):
        self.net.train()
        loss_vals = []
        for data, cofs, A, B, U in tqdm(self.train_dl, desc='Training Loop:', position=1, leave=False):
            error, loss_val = self.train_step(data, A, B, self.maxiter)
            loss_vals.append(loss_val)
            
        return np.array(loss_vals).mean()
    
    def val_loop(self):
        self.net.eval()
        real_loss_vals, jac_loss_vals = [], []

        for data, cofs, A, B, U in tqdm(self.val_dl, desc='Validation Loop:', position=2, leave=False):
            batch_size = U.shape[0]
            pre, jac_loss_val, real_loss_val = self.val_step(data, A, B, U, self.maxiter)

            real_loss_vals.append(real_loss_val)
            jac_loss_vals.append(jac_loss_val)

            self.writer.add_scalar("ValLoss", real_loss_val, self.val_global_idx)
            self.writer.add_scalar("ValJacLoss", jac_loss_val, self.val_global_idx)

            pre = pre.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
            sols = U.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)
            cofs = cofs.cpu().numpy().reshape(batch_size, self.GridSize, self.GridSize)

            k = np.random.choice(batch_size)
            fig = draw_img('Validation', pre[k], sols[k], cofs[k], self.GridSize, a=1)
            self.writer.add_figure(f"ValFigure", fig, self.val_global_idx)

            self.val_global_idx += 1

        return np.array(loss_vals).mean(), np.array(jac_loss_vals).mean()

if __name__ == '__main__':
    # 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    # GridSize = 192
    GridSize = 96
    mission_name = 'allcofs'
    tag = 'BACK_Ju'

    trainer = Trainer(
        area = ((0, 0), (1, 1)),
        GridSize=GridSize,
        trainN=10000,
        valN=100,
        net_kwargs={
            'model_name': 'segmodel',
            'Block': "ResBasic",
            'planes':6,
            'in_channels':3,
            'classes':1,
            'GridSize':GridSize,
            'layer_nums':[4, 6, 6, 8, 8],
            'adaptor_nums':[4, 6, 6, 8, 8],
            'factor':2,
            'norm_method': 'layer',
            'pool_method':'max',
            'padding':'same',
            'padding_mode':'replicate',
        },
        batch_size=20,
        method='jac',
        maxiter=20,
        log_dir=f'./all_logs/{mission_name}',
        lr=1e-3,
        loss_fn=torch.nn.functional.mse_loss,
        model_save_path=f'./model_save/{mission_name}',
        tag = tag,
        total_epochs=[150, 150],
        device='cuda',
        dtype=torch.float,
        hyper_params_save_path=f'./hyper_parameters/{mission_name}'
    )
    trainer.fit_loop()

    

        


