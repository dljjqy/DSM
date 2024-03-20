import torch
from torch.utils.tensorboard import SummaryWriter
from SegModel import CNN
from pathlib import Path
from tqdm import tqdm
import json

class BaseTrainer:
    def __init__(
            self,
            dtype, device,
            area, GridSize,
            trainN, valN, batch_size,
            net_kwargs, 
            log_dir, 
            lr=1e-3, total_epochs = [150],
            tag = '',
            loss_fn = torch.nn.functional.mse_loss, 
            model_save_path = './model_save', 
            hyper_params_save_path='./hyper_parameters/test',):
        
        self.dtype = dtype
        self.device = device

        self.area = area
        self.GridSize = GridSize

        self.trainN = trainN
        self.valN = valN
        self.batch_size = batch_size

        self.net_kwargs = net_kwargs.copy()
        self.init_network(net_kwargs)

        self.log_dir = log_dir
        self.lr = lr
        self.tag = tag
        self.total_epochs = total_epochs
        
        self.tag = tag
        self.loss_fn = loss_fn        
        
        self.model_save_path = f"{model_save_path}/{GridSize}/{self.name}"
        if not Path(self.model_save_path).is_dir():
            Path(self.model_save_path).mkdir(parents=True)

        self.hyper_param_save_path = f"{hyper_params_save_path}/{GridSize}"
        if not Path(self.hyper_param_save_path).is_dir():
            Path(self.hyper_param_save_path).mkdir(parents=True)

        self.train_global_idx = 0
        self.val_global_idx = 0
        self.global_epoch_idx=0
        self.best_train_loss = torch.inf
        self.best_val_real_loss = torch.inf
        self.best_val_subitr_loss = torch.inf

        self.save_hyper_parameters()
        self.init_tensorboard()

    @property
    def name(self):
        pass
    
    def hyper_param_need2save(self):
        param = {
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

    def save_hyper_parameters(self):
        with open(f"{self.hyper_param_save_path}/{self.name}.json", "w") as f:
            json.dump(self.hyper_param_need2save(), f)
        
    def init_tensorboard(self):
        self.writer = SummaryWriter(log_dir=f'{self.log_dir}/{self.name}', comment="Test")

    def init_network(self, kwargs):
        name = kwargs.pop('model_name')
        self.net = CNN(name, kwargs, self.dtype, self.device)

    def config_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.optimizer.zero_grad()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def reboot(self):
        self.init_traindl()
        self.init_valdl()
        self.config_optimizer(self.lr)

    def init_traindl(self):
        pass

    def init_valdl(self):
        pass
        
    def train_step(self, *args):
        pass

    def val_step(self, *args):
        pass
    
    def train_loop(self):
        pass
    
    def val_loop(self):
        pass

    def save_best_train(self, new_train_loss):
        if new_train_loss <= self.best_train_loss:
            self.best_train_loss = new_train_loss
            print(f"\nEpoch {self.global_epoch_idx} Best train loss: {self.best_train_loss:5<.3e} LR: {self.lr_scheduler.get_last_lr()[0]:5<.2e}")
            torch.save(self.net.state_dict(), f'{self.model_save_path}/best_train.pt')
    
    def save_best_val_real(self, new_val_real_loss):
        if new_val_real_loss <= self.best_val_real_loss:
            self.best_val_real_loss = new_val_real_loss
            print(f"\nEpoch {self.global_epoch_idx} Best validation  Real loss: {self.best_val_real_loss:5<.3e} LR: {self.lr_scheduler.get_last_lr()[0]:5<.2e}")
            torch.save(self.net.state_dict(), f'{self.model_save_path}/best_val.pt')
    
    def save_best_val_subitr(self, new_val_subitr_loss):
        if new_val_subitr_loss <= self.best_val_subitr_loss:
            self.best_val_subitr_loss = new_val_subitr_loss
            print(f"\nEpoch {self.global_epoch_idx} Best validation subitr loss: {self.best_val_subitr_loss:5<.3e} LR: {self.lr_scheduler.get_last_lr()[0]:5<.2e}")
            torch.save(self.net.state_dict(), f'{self.model_save_path}/best_val_itr.pt')

    def fit_loop(self):
        for epoch_num in self.total_epochs:
            self.reboot()

            for _ in tqdm(range(epoch_num), desc='Training Epoch:', leave=True):
                new_train_loss = self.train_loop()
                with torch.no_grad():
                    new_val_real_loss, new_val_subitr_loss = self.val_loop()

                self.lr_scheduler.step()
                self.global_epoch_idx += 1
                torch.save(self.net.state_dict(), f'{self.model_save_path}/last.pt')
                
                self.save_best_train(new_train_loss)
                self.save_best_val_real(new_val_real_loss)
                self.save_best_val_subitr(new_val_subitr_loss)

    