import torch
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import JacIter, PieceWiseConst, my_plot, coo2tensor, L2Loss, mmbv
import time 

class FvmVaryKDataset(Dataset):
    def __init__(self, GridSize, matrix_path, mus, b, device, index_start=0):
        h = 1 / GridSize
        x = np.arange(h/2, 1, h)
        y = np.arange(h/2, 1, h)
        self.xx, self.yy = np.meshgrid(x, y)

        self.matrix_path = matrix_path
        self.mus = mus
        self.b = torch.from_numpy(b).float()
        self.device = device        
        self.start = index_start

    def read_matrix(self, index):
        a = sparse.load_npz(f"{self.matrix_path}/A{self.start + index}.npz").tocoo()
        return coo2tensor(a, self.device)
     
    def __len__(self):
        return self.mus.shape[0]

    def __getitem__(self, index):
        B = self.b
        A = self.read_matrix(index)
        pwc = PieceWiseConst(self.mus[index])
        
        data = np.stack([self.xx, self.yy, pwc(self.xx, self.yy)], axis=0)
        data = torch.from_numpy(data).float()
        return A, data.to(self.device), B.to(self.device) 

class FvmVaryKValDataset(FvmVaryKDataset):
    def __init__(self, GridSize, matrix_path, mus, ans, b, device, index_start):
        super().__init__(GridSize, matrix_path, mus, b, device, index_start)
        self.GridSize = GridSize
        self.U = torch.from_numpy(ans).float()

    def __getitem__(self, index):
        B = self.b
        u = self.U[index].reshape(1, self.GridSize, self.GridSize)
        A = self.read_matrix(index)
        pwc = PieceWiseConst(self.mus[index])
        
        data = np.stack([self.xx, self.yy, pwc(self.xx, self.yy)], axis=0)
        data = torch.from_numpy(data).float()

        return A, data.to(self.device), B.to(self.device), u.to(self.device) 

class MyTrainer:
    def __init__(self, GridSize, t, trainN, valN, data_path, 
            encoder_name, batch_size, maxiter, log_dir, lr, 
            loss_fn = torch.nn.functional.mse_loss, 
            model_save_path = './model_save', 
            total_epochs=150):
        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = 'cpu'
            raise 'CUDA Initilize Failed!'
        
        self.Nx = GridSize
        self.BatchSize = batch_size
        self.loss_fn = loss_fn

        self.l2 = L2Loss(1/GridSize)

        self.maxiter = maxiter
        self.epochs = total_epochs
        self.lr = lr
        self.encoder_name = encoder_name
        self.exp_name = f"VaryK{t}x{t}-{GridSize}-{encoder_name}-{trainN}-{batch_size}"
        self.init_dataloader(data_path, t, GridSize, trainN, valN, batch_size)
        self.init_network(encoder_name)
        self.config_optimizer(lr)
        self.init_tensorboard(log_dir)
        self.save_path = f"{model_save_path}/{self.exp_name}"

        if not Path(self.save_path).is_dir():
            Path(self.save_path).mkdir(parents=True)

    def init_tensorboard(self, log_dir):
        self.writer = SummaryWriter(log_dir=f'{log_dir}/{self.exp_name}', comment="Test")

    def init_dataloader(self, data_path, t, GridSize, trainN, valN, batch_size):
        csv = pd.read_csv(f"{data_path}/square{t}x{t}.csv")
        mus = csv.values[:, 1:]
        ans = np.load(f"{data_path}/square{t}x{t}/GridSize-{GridSize}/U.npy")
        dataN = mus.shape[0]
        mus = mus.reshape((dataN, t, t))

        matrix_path = f"{data_path}/square{t}x{t}/GridSize-{GridSize}"
        b = np.load(f"{matrix_path}/b.npy")

        train_ds = FvmVaryKDataset(GridSize, matrix_path, mus[:trainN], b, self.device, 0)
        val_ds = FvmVaryKValDataset(GridSize, matrix_path, mus[trainN:trainN+valN], ans[trainN:trainN+valN], b, self.device, trainN)

        self.train_dl = DataLoader(train_ds, batch_size, True, )
        self.val_dl = DataLoader(val_ds, batch_size, True, )
        self.train_global_idx = 0
        self.val_global_idx = 0

    def init_network(self, encoder_name):
        self.net = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1)
        self.net = self.net.to(self.device)

    def config_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.optimizer.zero_grad()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    # def train_step(self, x, A, b, maxiter):
    #     pre = self.net(x)
    #     u = pre.reshape(self.BatchSize, -1)
    #     delta = torch.bmm(A, u[..., None]).squeeze() - b
    #     loss = self.loss_fn(delta, torch.zeros_like(delta))
    #     return loss
    
    # def val_step(self, x, A, b, u, maxiter):
    #     pre = self.net(x)
    #     x = pre.reshape(self.BatchSize, -1)
    #     delta = torch.bmm(A, x[..., None]).squeeze() - b
    #     val_resloss = self.loss_fn(delta, torch.zeros_like(delta))
    #     val_loss = self.l2(pre, u)
    #     return pre, val_resloss, val_loss
    
    def train_step(self, x, A, b, maxiter):
        pre = self.net(x)

        with torch.no_grad():
            generator = JacIter(A, maxiter)
            ans = generator(pre, b)
        
        loss = self.loss_fn(pre, ans)
        return loss
    
    def val_step(self, x, A, b, u, maxiter):
        pre = self.net(x)

        generator = JacIter(A, maxiter)
        jac_ans = generator(pre, b)
        
        val_jac_loss = self.loss_fn(jac_ans, pre)
        val_loss = self.l2(pre, u)
        return pre, val_jac_loss, val_loss

    def train_loop(self):
        self.net.train()
        loss_vals = []

        with tqdm(enumerate(self.train_dl), total=len(self.train_dl), position=0) as train_bar:
            for matrixA, x, b in self.train_dl:
                loss_val = self.train_step(x, matrixA, b, self.maxiter)

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
            for idx, (matrixA, x, b, u) in enumerate(self.val_dl):
                pre, jac_loss_val, loss_val = self.val_step(x, matrixA, b, u, self.maxiter)
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
    "GridSize" : 64,
    "t" : 2,
    "trainN" : 9000,
    "valN" : 1000,
    "data_path" : './DLdata',
    "encoder_name" : 'resnet50',
    "batch_size" : 10,
    "maxiter" : 5,
    "log_dir" : './all_logs/varyk-res',
    "lr" : 1e-3,
    "loss_fn" : torch.nn.functional.mse_loss,
    "model_save_path" : './model_save/varyk-res',
    "total_epochs" : [100, 200],
    }
    trainer = MyTrainer(**kwargs)
    print(trainer.exp_name)
    trainer.fit_loop()

    

        


