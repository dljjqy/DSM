import sys
sys.path.append('../')

import torch
import numpy as np
import torch.nn.functional as F
from utils import L2Loss, kappa, force
from MyPlot import multi_nonlinear_draw_img as draw_img
from tqdm import tqdm
from random import uniform
from torch.utils.data import DataLoader, Dataset
from BaseTrainer import BaseTrainer
from itertools import product


def hard_encode(x, gd):
    y = F.pad(x, (1, 1, 1, 1), "constant", value=gd)
    return y

class C2MuTrainDs(Dataset):
    def __init__(self, GridSize, dtype, device, trainN, area=((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize), np.linspace(bottom, top, GridSize)
        )
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)

        center_points = np.random.uniform(0.05, 0.95, (trainN, 2))
        self.center_points = torch.from_numpy(center_points)
        self.area = area
        self.trainN = trainN

    def __len__(self):
        return self.trainN

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)
        mu = torch.ones_like(f) * uniform(0.1, 1)
        data = torch.stack([f, mu]).to(self.dtype).to(self.device)
        return data, f[None, ...], mu[None, ...]


class C2MuValDs(Dataset):
    def __init__(self, GridSize, dtype, device, area=((0, 0), (1, 1))):
        self.dtype = dtype
        self.device = device
        (left, bottom), (right, top) = area
        xx, yy = np.meshgrid(
            np.linspace(left, right, GridSize), np.linspace(bottom, top, GridSize)
        )
        self.xx, self.yy = torch.from_numpy(xx), torch.from_numpy(yy)
        self.center_points = list(
            product(torch.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1))
        )
        self.area = area

        U = np.load(f"DLdata/{GridSize}/U.npy")
        self.U = torch.from_numpy(U).float()

    def __len__(self):
        return len(self.center_points)

    def __getitem__(self, index):
        f = force(self.xx, self.yy, self.center_points[index]).to(self.dtype).to(self.device)
        mu = torch.ones_like(f) * 0.1
        data = torch.stack([f, mu]).to(self.dtype).to(self.device)

        ans = self.U[index].to(self.dtype).to(self.device)
        return data, f[None, ...], mu[None, ...], ans[None, ...]


class PinnGenerator(torch.nn.Module):
    def __init__(
        self, batch_size, GridSize, dtype, device, maxiter, area, init_kappa=None, gd=0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.GridSize = GridSize
        (left, bottom), (right, top) = area
        self.h = (right - left) / (GridSize - 1)
        self.dtype = dtype
        self.device = device
        self.maxiter = maxiter
        self.gd = gd
        self.hard_encode = lambda x: hard_encode(x, self.gd)

        self.k1 = self._get_kernel([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
        self.k2 = self._get_kernel([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]])
        self.k3 = self._get_kernel([[0, 0.5, 0], [0.5, 2, 0.5], [0, 0.5, 0]])

        self.prev_pre = init_kappa

    def _get_kernel(self, k):
        k = torch.tensor(k, requires_grad=True)
        k = k.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(self.dtype).to(self.device)
        return k

    def jac_step(self, pre, f, mu):
        u = self.hard_encode(pre)
        w = kappa(self.prev_pre, mu)

        force = f[..., 1:-1, 1:-1] * self.h**2
        y1 = F.conv2d(u, self.k1) * w[..., 1:-1, 1:-1]
        y2 = F.conv2d(w * u, self.k2)
        y3 = F.conv2d(w, self.k3)
        return (force + y1 + y2) / y3

    def forward(self, pre, f, mu):
        with torch.no_grad():
            y = self.jac_step(pre, f, mu)
            for _ in range(self.maxiter):
                y = self.jac_step(y, f, mu)
        return y


class NConvTrainer(BaseTrainer):
    def __init__(
        self,
        gd=0,
        maxiter=5,
        picard_eps=1e-5,
        subitr_eps=1e-7,
        max_subitr_step=500,
        max_picard_step=30,
        *args,
        **kwargs,
    ):
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
        self.global_data_nums = 0
        # self.init_generator()

    @property
    def name(self):
        return f"{self.tag}-{self.net.name()}-GridSize:{self.GridSize}-maxiter:{self.maxiter}-trainN:{self.trainN}-bs:{self.batch_size}"

    def hyper_param_need2save(self):
        param = {
            "gd": self.gd,
            "maxiter": self.maxiter,
            "picard_eps": self.picard_eps,
            "subitr_eps": self.subitr_eps,
            "max_subitr_step": self.max_subitr_step,
            "nax_picard_step": self.max_picard_step,
            "GridSize": self.GridSize,
            "area": self.area,
            "trainN": self.trainN,
            "valN": self.valN,
            "Batch_size": self.batch_size,
            "lr": self.lr,
            "epochs": self.total_epochs,
            "tag": self.tag,
            "net_kwargs": self.net_kwargs,
        }
        return param

    def init_traindl(self):
        train_ds = C2MuTrainDs(
            self.GridSize, self.dtype, self.device, self.trainN, self.area
        )
        self.train_dl = DataLoader(train_ds, self.batch_size, shuffle=True)

    def init_valdl(self):
        val_ds = C2MuValDs(self.GridSize, self.dtype, self.device, self.area)
        self.val_dl = DataLoader(val_ds, 1, shuffle=True)

    def reboot(self):
        self.init_traindl()
        self.init_valdl()
        self.config_optimizer(self.lr)

    def epoch_reboot(self):
        pass

    def init_generator(self, w=None):
        generator = PinnGenerator(
            self.batch_size,
            self.GridSize,
            self.dtype,
            self.device,
            self.maxiter,
            self.area,
            w,
            self.gd,
        )
        return generator

    def val_plot(self, u, f, ans, name):
        u = u.reshape(self.GridSize, self.GridSize)
        u = u.detach().cpu().numpy()

        force = f.reshape(self.GridSize, self.GridSize)
        force = force.detach().cpu().numpy()

        ans = ans.reshape(self.GridSize, self.GridSize)
        ans = ans.detach().cpu().numpy()

        fig = draw_img("Validation", force, 0.1, u, ans, self.GridSize, a=1.0)
        return fig

    def train_step(self, x, f, mu, generator):
        # Given the generator <==> Given a linear system.
        # One train step is for let Network approximate the solution of linear system better.
        pre = self.net(x)
        with torch.no_grad():
            label = generator(torch.clone(torch.detach(pre)), f, mu)

        train_loss = self.loss_fn(pre, label)

        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        return pre, label, train_loss

    def picard_train_step(self, x, f, mu, generator):
        # The single one step of picard iteration is used for solving a linear PDE
        # The linear sub-iteration JAC is used to optimize the network
        for i in tqdm(
            range(self.max_subitr_step),
            position=3,
            leave=False,
            desc="Linear Sub-Iteration:",
        ):
            pre, label, train_loss = self.train_step(x, f, mu, generator)

            with torch.no_grad():
                self.train_global_idx += 1
                self.writer.add_scalar(
                    "Train-InnerLoopLoss", train_loss.item(), self.train_global_idx
                )

                subitr_error = self.l2_loss(pre, label).item()
                if subitr_error < self.subitr_eps:
                    break
        return hard_encode(pre, self.gd), i+1

    def picard_loop(self, x, f, mu, old_pre):
        generator = self.init_generator(old_pre)
        inner_loop_steps = []
            
        for i in tqdm(
            range(self.max_picard_step),
            position=2,
            leave=False,
            desc="Picard Iteration:",
            ):
            new_pre, inner_loop_step = self.picard_train_step(x, f, mu, generator)
            inner_loop_steps.append(inner_loop_step)
            with torch.no_grad():
                picard_error = self.l2_loss(old_pre, new_pre).item()

                pre = torch.clone(torch.detach(new_pre))
                generator = self.init_generator(pre)
                pre = pre[..., 1:-1, 1:-1]
                label = generator(pre, f, mu)

                picard_loss = self.loss_fn(label, pre).item()

                self.writer.add_scalar(
                    "Train-PicardL2", picard_error, self.picard_global_step
                )
                self.writer.add_scalar(
                    "Train-PicardMSE", picard_loss, self.picard_global_step
                )
                
                self.picard_global_step += 1

                old_pre = new_pre
                # if picard_error <= self.picard_eps or picard_loss <= self.picard_eps:
                #     break
                
        return picard_error, i+1, sum(inner_loop_steps)

    def train_loop(self):
        self.net.train()
        train_picard_loss = []

        for x, f, mu in tqdm(
            self.train_dl, position=1, leave=True, desc="Training Loop:"
        ):
            # Initialize the diffusion parameter for the next picard loop
            with torch.no_grad():
                if self.global_epoch_idx == 0:
                    old_pre = (
                        torch.ones((self.batch_size, 1, self.GridSize, self.GridSize)).to(self.dtype).to(self.device)
                    )
                else:
                    old_pre = torch.clone(
                        torch.detach(hard_encode(self.net(x), self.gd))
                    )

            # Start fitting the next picard loop by given the data, force and mu
            picard_loss, picard_iteration_steps, summed_inner_loop_steps = self.picard_loop(x, f, mu, old_pre=old_pre)
            train_picard_loss.append(picard_loss)
            self.writer.add_scalar(
                "Train-PicardIterationSteps", picard_iteration_steps, self.global_data_nums
            )
            self.writer.add_scalar(
                "Train-SummedInnerLoopSteps", summed_inner_loop_steps, self.global_data_nums
            )

            self.global_data_nums += 1
        
        self.lr_scheduler.step()
        return np.array(train_picard_loss).mean()

    def val_step(self, x, f, mu, ans):
        pre = self.net(x)

        generator = self.init_generator(w=hard_encode(pre, self.gd))
        label = generator(pre, f, mu)

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
            for x, f, mu, ans in self.val_dl:
                pre, val_loss, real_loss = self.val_step(x, f, mu, ans)

                val_subitr_loss.append(val_loss)
                val_real_loss.append(real_loss)
                self.val_global_idx += 1

            fig = self.val_plot(
                u=hard_encode(pre, self.gd),
                f=f,
                ans=ans,
                name=f"Val-{self.val_global_idx}",
            )

            self.writer.add_figure("ValFigure", fig, self.val_global_idx)
            return np.array(val_subitr_loss).mean(), np.array(val_real_loss).mean()


if __name__ == "__main__":
    GridSize = 128
    max_inner_loop_step = 1

    max_picard_step = 1
    
    tag = f"Picard={max_picard_step}"
    trainer = NConvTrainer(
        gd=0,
        maxiter=10,
        picard_eps=1e-8,
        subitr_eps=5e-9,
        max_subitr_step=max_inner_loop_step,
        max_picard_step=max_picard_step,
        dtype=torch.float,
        device="cuda",
        area=((0, 0), (1, 1)),
        GridSize=GridSize,
        trainN=10000,
        valN=100,
        batch_size=5,
        net_kwargs={
            "model_name": "segmodel",
            "Block": "ResBottleNeck",
            "planes": 8,
            "in_channels": 2,
            "classes": 1,
            "GridSize": GridSize,
            "layer_nums": [4, 4, 6, 6, 8],
            "adaptor_nums": [4, 4, 6, 6, 8],
            "factor": 2,
            "norm_method": "layer",
            "pool_method": "max",
            "padding": "same",
            "padding_mode": "zeros",
            "end_padding_mode": "zeros",
            "end_padding": "valid",
            "act":"tanh"
        },
        log_dir=f"./all_logs",
        lr=1e-3,
        total_epochs=[150],
        tag=tag,
        loss_fn=F.mse_loss,
        model_save_path=f"./model_save",
        hyper_params_save_path=f"./hyper_parameters",
    )
    trainer.fit_loop()
