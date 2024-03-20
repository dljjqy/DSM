import torch
import numpy as np
# import segmentation_models_pytorch as smp
from MyPlot import my_plot
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils import *
from varyUNet import VaryUNet
from itertools import product


# torch.set_default_tensor_type(torch.DoubleTensor)
def hard_encode(x, gd):
    y = F.pad(x, (1, 1, 1, 1), 'constant', value=gd)
    return y

def kappa(u, mu=0.5):
    return ((3 * u * (1 - u)) / (3 * u**3 + (1-u)**3) )**2 + mu

def force(x, y, center, delta=0.05):
    px, py = center
    mask = (x > px-delta) * (x <= px+delta) * (y > py-delta) * (y <= py+delta)
    force = 100 * np.exp(-50 * ((x - px)**2 + (y - py)**2))
    return mask * force


class TrainDs(Dataset):
    def __init__(self, GridSize, dtype, device, trainN, area = ((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize),
            np.linspace(bottom, top, GridSize)
        ) 
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)

        center_points = np.random.uniform(0.1, 0.9, (trainN, 2))
        self.center_points = torch.from_numpy(center_points)
        self.area = area
        self.trainN = trainN

    def __len__(self):
        return self.trainN

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)
        data = torch.clone(f[None, ...])
        return data, f[None, ...]

class ValDs(Dataset):
    def __init__(self, GridSize, dtype, device, area = ((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize),
            np.linspace(bottom, top, GridSize)
        ) 
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)
        self.center_points = list(product(torch.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1)))
        self.area = area

        U = np.load(f'DLdata/nlinear/{GridSize}/U.npy')
        self.U = torch.from_numpy(U).float()

    def __len__(self):
        return len(self.center_points)

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)
        data = torch.clone(f[None, ...])
        # data = torch.stack([self.xx, self.yy, f], dim=0).to(self.dtype).to(self.device)
        ans = self.U[index].to(self.dtype).to(self.device)
        return data, f[None, ...], ans[None, ...]

class PinnGenerator(torch.nn.Module):
    def __init__(self, GridSize, device, maxiter, area, prev_net=None, gd=0):
        super().__init__()
        self.GridSize = GridSize
        (left, bottom), (right, top) = area
        self.h = (right - left) / (GridSize - 1)
        self.device = device
        self.maxiter = maxiter
        self.gd = gd

        self.k1 = self._get_kernel([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
        self.k2 = self._get_kernel([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
        self.k3 = self._get_kernel([[0, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 0]])

        self.prev_net = prev_net

        self.hard_encode = lambda x: hard_encode(x, self.gd)

    def _get_kernel(self, k):
        k = torch.tensor(k, requires_grad=True)
        k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).float().to(self.device)
        return k

    def jac_step(self, x, pre, f, mu):
        u = self.hard_encode(pre)
        if self.prev_net is None:
            w = torch.ones_like(f, requires_grad=False)
        else:
            w = self.hard_encode(self.prev_net(x))
        
        w = kappa(w, mu)
        force = f[..., 1:-1, 1:-1] * self.h**2
        y1 = F.conv2d(u, self.k1) * w[..., 1: -1, 1: -1]        
        y2 = F.conv2d(w * u, self.k2)
        y3 = F.conv2d(w, self.k3)
        return (force + y1 + y2) / y3
    
    def forward(self, x, pre, f, mu):
        with torch.no_grad():
            y = self.jac_step(x, pre, f, mu)
            for _ in range(self.maxiter):
                y = self.jac_step(x, y, f, mu)
        return y
    
class NConvTrainer:
    def __init__(
            self, 
            GridSize, 
            encoder_name, 
            gd, 
            dtype,
            mu,
            trainN, maxiter=5, epochs = [100], loss_fn = F.mse_loss,
            area=((0, 0), (1, 1)), batch_size=1, lr=1e-3, 
            log_dir='./all_logs/nlinear_conv',
            save_dir = './model_save/nlinear_conv'):
        self.GridSize = GridSize
        self.encoder_name = encoder_name
        self.gd = gd
        self.dtype = dtype
        self.mu = mu
        self.trainN = trainN
        self.maxiter = maxiter
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.area = area
        self.batch_size = batch_size
        self.lr = lr
        self.max_picard_step = 20
        self.max_step = 100
        self.h = 1.0 / (GridSize - 1)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            raise ValueError("Cuda Filed")
        
        self.exp_name = f"{encoder_name}_{GridSize}_{trainN}x{self.epochs[0]}x{len(self.epochs)}_{maxiter}"
        self.save_path = f"{save_dir}/{self.exp_name}"
        if not Path(self.save_path).is_dir():
            Path(self.save_path).mkdir(parents=True)

        self.init_tensorboard(log_dir=log_dir)

        self.train_global_idx = 0
        self.val_global_idx = 0
        self.picard_global_idx = 0
        self.prev_net = None
        self.best_real_loss = torch.inf

    def tensor_l2(self, x, y):
        diff = (((x - y) ** 2) * self.h**2)
        error = diff.sum()
        return error

    def init_dataloader(self, trainN):
        train_ds = TrainDs(self.GridSize, self.dtype, self.device, trainN, self.area)
        val_ds = ValDs(self.GridSize, self.dtype, self.device, self.area)

        train_dl = DataLoader(train_ds, self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, 1, shuffle=True)
        return train_dl, val_dl
    
    def init_generator(self, prev_net):
        self.generator = PinnGenerator(self.GridSize, self.device, self.maxiter, self.area, prev_net, self.gd)
        
    def init_tensorboard(self, log_dir):
        self.writer = SummaryWriter(log_dir=f'{log_dir}/{self.exp_name}', comment="Test")

    def init_network(self):
        net = VaryUNet(1, 1, 16, 5, end_padding='valid')
        net = net.to(self.device)
        return net
    
    def config_optimizer(self, net, lr):
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr,)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
    def val_plot(self, u, f, ans, name):
        u = u.reshape(self.GridSize, self.GridSize)
        u = u.detach().cpu().numpy()
        
        force = f.reshape(self.GridSize, self.GridSize)
        force = force.detach().cpu().numpy()

        ans = ans.reshape(self.GridSize, self.GridSize)
        ans = ans.detach().cpu().numpy()

        fig = my_plot(self.GridSize, u, ans, force, name, 1.0)
        return fig
    
    def train_step(self, net, x, f):
        pre = net(x)
        with torch.no_grad():
            label = self.generator(x.detach(), pre, f.detach(), self.mu)
        train_loss = self.loss_fn(pre, label)
        self.writer.add_scalar("TrainLoss", train_loss.item(), self.train_global_idx)

        return train_loss

    def train_loop(self, train_dl, net):
        for x, f in tqdm(train_dl, position=1, leave=False):
            self.optimizer.zero_grad()
            train_loss = self.train_step(net, x, f)

            train_loss.backward()
            self.optimizer.step()

            self.train_global_idx += 1

        self.lr_scheduler.step()
        torch.save(net, f"{self.save_path}/last_train.pt")


    def val_step(self, net, x, f, ans):
        pre = net(x)
        label = self.generator(x, pre, f, self.mu) 
        val_loss = self.loss_fn(pre, label)
        real_loss = self.loss_fn(hard_encode(pre, self.gd), ans)

        self.writer.add_scalar("ValLoss", val_loss.item(), self.val_global_idx)
        self.writer.add_scalar("ValRealLoss", real_loss.item(), self.val_global_idx)

        return pre, val_loss.item(), real_loss.item()

    def val_loop(self, net, val_dl):
        monitor_real_loss = []
        with torch.no_grad():
            net.eval()
            for x, f, ans in val_dl:
                pre, val_loss, real_loss = self.val_step(net, x, f, ans)
                
                monitor_real_loss.append(real_loss)
                self.val_global_idx += 1

        fig = self.val_plot(
            u = hard_encode(pre, self.gd),
            f = f,
            ans = ans,
            name = f"Val-{self.val_global_idx}")
            
        self.writer.add_figure("ValFigure", fig, self.val_global_idx)

        new_real_loss = sum(monitor_real_loss)
        if new_real_loss <= self.best_real_loss:
            torch.save(net, f"{self.save_path}/best_val.pt")

    def picard_step(self, epoch, prev_net=None):
        train_dl, val_dl = self.init_dataloader(self.trainN)
        net = self.init_network()
        self.config_optimizer(net, self.lr)
        self.init_generator(prev_net)
        
        for _ in tqdm(range(epoch), position=0, leave=True, 
                    desc=f"Picard Step: {self.picard_global_idx}"):
            net.train()
            self.train_loop(train_dl, net)
            self.val_loop(net, val_dl)

        self.picard_global_idx += 1
        torch.save(net, f"{self.save_path}/last_picard.pt")

        return net
                
    def fit(self):
        prev_net = self.picard_step(self.epochs[0], None)
        for i in range(1, len(self.epochs)):
            prev_net = self.picard_step(self.epochs[i], prev_net)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float)
    trainer = NConvTrainer(
        GridSize=64,
        encoder_name='maskF',
        gd=0,
        dtype=torch.float,
        mu=0.1,
        trainN=2000,
        maxiter=5,
        epochs=[150] * 20,
        loss_fn=F.mse_loss,
        area = ((0, 0), (1, 1)),
        batch_size=10,
        lr = 1e-3,
        log_dir='./all_logs/test',
        save_dir='./model_save/test'
    )
    trainer.fit()