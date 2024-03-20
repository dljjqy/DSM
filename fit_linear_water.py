import torch
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *

class FvmLinWaterTrainDs(Dataset):
    def __init__(self, case, GridSize, trainN, device, data_path = './DLdata/water',):
        F = np.load(f"{data_path}/GridSize-{GridSize}/F.npy")
        b = np.load(f"{data_path}/GridSize-{GridSize}/case{case}/b.npy")

        self.F = torch.tensor(np.copy(F[:trainN])[:, np.newaxis, :]).float()
        self.b = torch.from_numpy(b).float()

        self.device = device
        self.h = 500.0 / GridSize
        x = np.arange(-250 + self.h/2, 250, self.h)
        y = np.arange(-250 + self.h/2, 250, self.h)
        xx, yy = np.meshgrid(x, y) 
        self.xx, self.yy = torch.from_numpy(xx).float(), torch.from_numpy(yy).float()
        del F, b
    
    def __len__(self):
        return self.F.shape[0]

    def __getitem__(self, index):
        f = self.F[index]
        b = self.b + torch.flatten(f) * self.h**2
        data = torch.stack([self.xx, self.yy, f.squeeze()], dim=0)
        return data.to(self.device), b.to(self.device) 

class FvmLinWaterValDs(Dataset):
    def __init__(self, case, GridSize, trainN, ValN,  device, data_path = './DLdata/water',):
        F = np.load(f"{data_path}/GridSize-{GridSize}/F.npy")
        b = np.load(f"{data_path}/GridSize-{GridSize}/case{case}/b.npy")
        U = np.load(f"{data_path}/GridSize-{GridSize}/case{case}/U.npy")

        self.F = torch.tensor(np.copy(F[trainN: trainN+ValN])[:, np.newaxis, :]).float()
        self.U = torch.tensor(np.copy(U[trainN: trainN+ValN])[:, np.newaxis, :]).float()
        self.b = torch.from_numpy(b).float()

        self.device = device
        self.h = 500.0 / GridSize
        x = np.arange(-250 + self.h/2, 250, self.h)
        y = np.arange(-250 + self.h/2, 250, self.h)
        xx, yy = np.meshgrid(x, y) 
        self.xx, self.yy = torch.from_numpy(xx).float(), torch.from_numpy(yy).float()
        del F, b
    
    def __len__(self):
        return self.F.shape[0]

    def __getitem__(self, index):
        f = self.F[index]
        u = self.U[index]
        b = self.b + torch.flatten(f) * self.h**2
        data = torch.stack([self.xx, self.yy, f.squeeze()], dim=0)
        return data.to(self.device), b.to(self.device), u.to(self.device)
     
class MyTrainer:
    def __init__(
            self, 
            bd_case, 
            tag, 
            GridSize, 
            method,
            trainN, valN, data_path, 
            encoder_name, batch_size, maxiter, log_dir, lr, 
            loss_fn = torch.nn.functional.mse_loss, 
            model_save_path = './model_save', 
            total_epochs = [150]):
            
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = 'cpu'
            raise 'CUDA Initilize Failed!'
        
        self.bd_case = bd_case
        self.tag =tag
        self.Nx = GridSize
        self.BatchSize = batch_size
        self.loss_fn = loss_fn
        self.maxiter = maxiter
        self.epochs = total_epochs
        self.lr = lr
        self.encoder_name = encoder_name
        self.exp_name = f"Case{bd_case}_{encoder_name}_{GridSize}_{tag}_{sum(self.epochs)}_{batch_size}_{method}-{maxiter}"
        self.generator = SubItr(method, maxiter, f"./DLdata/water/GridSize-{GridSize}/case{bd_case}/A.npz" , self.device)
        
        self.init_dataloader(bd_case, f"{data_path}", GridSize, trainN, valN, batch_size)
        self.init_network(encoder_name)
        self.config_optimizer(lr)
        self.init_tensorboard(log_dir)
        
        self.save_path = f"{model_save_path}/{self.exp_name}"
        if not Path(self.save_path).is_dir():
            Path(self.save_path).mkdir(parents=True)

    def init_tensorboard(self, log_dir):
        self.writer = SummaryWriter(log_dir=f'{log_dir}/{self.exp_name}', comment="Test")

    def init_dataloader(self, case, data_path, GridSize, trainN, valN, batch_size):
        train_ds = FvmLinWaterTrainDs(case, GridSize, trainN, self.device, data_path)
        val_ds = FvmLinWaterValDs(case, GridSize, trainN, valN, self.device, data_path)

        self.train_dl = DataLoader(train_ds, batch_size, True)
        self.val_dl = DataLoader(val_ds, batch_size, False)
        self.train_global_idx = 0
        self.val_global_idx = 0

    def init_network(self, encoder_name):
        self.net = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1)
        self.net = self.net.cuda()

    def config_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.optimizer.zero_grad()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def train_step(self, x, b, maxiter):
        self.generator.fresh_maxiter(maxiter)
        pre = self.net(x)
        with torch.no_grad():
            ans = self.generator(pre, b)
        
        loss = self.loss_fn(pre, ans)
        return loss
    
    def val_step(self, x, b, u, maxiter):
        self.generator.fresh_maxiter(maxiter)
        pre = self.net(x)

        jac_ans = self.generator(pre, b)
        
        val_jac_loss = self.loss_fn(jac_ans, pre)
        val_loss = self.loss_fn(pre, u)
        return pre, val_jac_loss, val_loss

    def train_loop(self):
        self.net.train()
        loss_vals = []

        with tqdm(enumerate(self.train_dl), total=len(self.train_dl), position=0) as train_bar:
            for x, b in self.train_dl:
                loss_val = self.train_step(x, b, self.maxiter)
                loss_val.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.writer.add_scalar("TrainJacLoss", loss_val.item(), self.train_global_idx)
                self.train_global_idx += 1
                loss_vals.append(loss_val.item())
                train_bar.update(1)
        return np.array(loss_vals).mean()
    
    def val_loop(self):
        self.net.eval()
        loss_vals = []
        jac_loss_vals = []

        with tqdm(enumerate(self.val_dl), total=len(self.val_dl), position=0) as val_bar:
            for idx, (x, b, u) in enumerate(self.val_dl):
                pre, jac_loss_val, loss_val = self.val_step(x, b, u, self.maxiter)
                self.val_global_idx += 1

                loss_val, jac_loss_val = loss_val.item(), jac_loss_val.item()

                loss_vals.append(loss_val)
                jac_loss_vals.append(jac_loss_val)

                val_bar.update(1)
                self.writer.add_scalar("ValLoss", loss_val, self.val_global_idx)
                self.writer.add_scalar("ValJacLoss", jac_loss_val, self.val_global_idx)
                
                if idx == 0:
                    BatchSize = pre.shape[0]
                    pre = pre.detach().cpu().numpy().reshape(BatchSize, self.Nx, self.Nx)
                    ans = u.detach().cpu().numpy().reshape(BatchSize, self.Nx, self.Nx)
                    x = x.detach().cpu().numpy().squeeze().reshape(BatchSize, 3, self.Nx, self.Nx)
                    kappa = x[0][-1]
                    fig = my_plot(self.Nx, pre[0], ans[0], kappa, 'Validation', a=1)
                    self.writer.add_figure("ValFigure", fig, self.val_global_idx)

        return np.array(loss_vals).mean(), np.array(jac_loss_vals).mean()

    def fit_loop(self):
        best_train_loss = np.inf
        best_val_loss = np.inf
        best_val_jac_loss = np.inf

        for i, epoch_num in enumerate(self.epochs):
            self.config_optimizer(self.lr)
            for epoch in range(epoch_num):
                new_train_loss = self.train_loop()
                with torch.no_grad():
                    new_val_loss, new_val_jac_loss = self.val_loop()
            
                if new_val_loss <= best_val_loss:
                    best_val_loss = new_val_loss
                    print(f"\nEpoch {epoch} Best valdation loss: {best_val_loss:5<.3e} LR: {self.lr_scheduler.get_last_lr()[0]:5<.2e}")
                    torch.save(self.net.state_dict(), f'{self.save_path}/best_val.pt')
                
                if new_val_jac_loss <= best_val_jac_loss:
                    best_val_jac_loss = new_val_jac_loss
                    print(f"\nEpoch {epoch} Best valdation loss: {best_val_jac_loss:5<.3e} LR: {self.lr_scheduler.get_last_lr()[0]:5<.2e}")
                    torch.save(self.net.state_dict(), f'{self.save_path}/best_jac_val.pt')

                if new_train_loss <= best_train_loss:
                    best_train_loss = new_train_loss
                    print(f"\nEpoch {epoch} Best train loss: {best_train_loss:5<.3e} LR: {self.lr_scheduler.get_last_lr()[0]:5<.2e}")
                    torch.save(self.net.state_dict(), f'{self.save_path}/best_train.pt')
            
                self.lr_scheduler.step()
                torch.save(self.net.state_dict(), f'{self.save_path}/last.pt')

  
if __name__ == '__main__':
    # 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    
    kwargs = {
        "bd_case" : 2,
        "tag": 'Reboot',
        "GridSize" : 128, 
        "method": "jac",
        "trainN" : 9000,
        "valN" : 1000,
        "data_path" : './DLdata/water',
        "encoder_name" : 'resnet18',
        "batch_size" : 10,
        "maxiter" : 5,
        "log_dir" : f'./all_logs/water',
        "lr" : 1e-3,
        "loss_fn" : torch.nn.functional.mse_loss,
        "model_save_path" : f'./model_save/water',
        "total_epochs" : [150],
    }
    trainer = MyTrainer(**kwargs)
    print(trainer.exp_name)
    trainer.fit_loop()