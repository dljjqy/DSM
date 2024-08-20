import torch
from torch.utils.tensorboard import SummaryWriter
from SegModel import CNN
from pathlib import Path
from tqdm import tqdm
import json

class BaseTrainer:
    def __init__(
            self,
            dtype: torch.dtype, 
            device: torch.device,
            area: any,  # Specify the type if known
            GridSize: int,
            trainN: int, 
            valN: int, 
            batch_size: int,
            net_kwargs: dict, 
            log_dir: str, 
            lr: float = 1e-3, 
            total_epochs: list = [150],
            tag: str = '',
            loss_fn = torch.nn.functional.mse_loss, 
            model_save_path: str = './model_save', 
            hyper_params_save_path: str = './hyper_parameters/test'):

        # Initialize parameters using setattr in a loop
        params = locals()
        for name, value in params.items():
            if name != 'self':
                setattr(self, name, value)
        
        self.net_kwargs = net_kwargs.copy()
        self.init_network(net_kwargs)
        
        self.model_save_path = f"{model_save_path}/{GridSize}/{self.name}"
        if not Path(self.model_save_path).is_dir():
            Path(self.model_save_path).mkdir(parents=True)

        self.hyper_param_save_path = f"{hyper_params_save_path}/{GridSize}"
        if not Path(self.hyper_param_save_path).is_dir():
            Path(self.hyper_param_save_path).mkdir(parents=True)

        self.train_global_idx = 0
        self.val_global_idx = 0
        self.global_epoch_idx=0
        self.best_train_error = torch.inf
        self.best_val_real_loss = torch.inf
        self.best_val_error = torch.inf

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
        net_kwargs = kwargs.copy()
        name = net_kwargs.pop('model_name')
        net = CNN(name, net_kwargs, self.dtype, self.device)
        self.net = net.to(self.dtype)
        
    def config_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.optimizer.zero_grad()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def reboot(self):
        pass

    def epoch_reboot(self):
        pass

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

    def save_best_model(self, new_value, best_value_attr, file_name):
        if new_value <= getattr(self, best_value_attr):
            setattr(self, best_value_attr, new_value)
            torch.save(self.net.state_dict(), f'{self.model_save_path}/{file_name}')
    
    def save_best_train_error(self, new_train_error):
        self.save_best_model(new_train_error, 'best_train_error', 'best_train.pt')
    
    def save_best_val_real(self, new_val_real_loss):
        self.save_best_model(new_val_real_loss, 'best_val_real_loss', 'best_val.pt')
    
    def save_best_val_error(self, new_val_error):
        self.save_best_model(new_val_error, 'best_val_error', 'best_val_itr.pt')


    def fit_loop(self):
        for epoch_num in self.total_epochs:
            self.reboot()

            for _ in tqdm(range(epoch_num), desc='Training Epoch:', leave=True):
                self.epoch_reboot()

                new_train_error = self.train_loop()
                with torch.no_grad():
                    new_val_real_loss, new_val_error = self.val_loop()

                self.lr_scheduler.step()
                self.global_epoch_idx += 1
                torch.save(self.net.state_dict(), f'{self.model_save_path}/last.pt')
                
                self.save_best_train_error(new_train_error)
                self.save_best_val_real(new_val_real_loss)
                self.save_best_val_error(new_val_error)


    