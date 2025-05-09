import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, List
from typing import Type, Union
# import segmentation_models_pytorch as smp

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        GridSize: int = 64,
        norm_method: Literal['batch', 'layer'] = 'batch', 
        padding='same',
        padding_mode='reflect',):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=padding, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=padding, padding_mode=padding_mode)
        # self.act = nn.ReLU(inplace=True)

        match norm_method:
            case 'batch':
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
            case 'layer':
                self.bn1 = nn.LayerNorm([out_channels, GridSize, GridSize])
                self.bn2 = nn.LayerNorm([out_channels, GridSize, GridSize])
            case _:
                self.bn1 = nn.Identity()
                self.bn2 = nn.Identity()
        

        self.block = nn.Sequential(
            self.conv1, 
            self.bn1,
            nn.ReLU(),
            self.conv2,
            self.bn2,
            nn.ReLU()
        )

    def forward(self, x):
        out = self.block(x)
        return out

class ResBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        GridSize: int = 64,
        norm_method: Literal['batch', 'layer'] = 'batch', 
        padding='same',
        padding_mode='reflect',
        act=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=padding, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=padding, padding_mode=padding_mode)
        self.act = act

        match norm_method:
            case 'batch':
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
            case 'layer':
                self.bn1 = nn.LayerNorm([out_channels, GridSize, GridSize])
                self.bn2 = nn.LayerNorm([out_channels, GridSize, GridSize])
            case _:
                self.bn1 = nn.Identity()
                self.bn2 = nn.Identity()
        

        self.block = nn.Sequential(
            self.conv1, 
            self.bn1,
            self.act,
            self.conv2,
            self.bn2,
        )

        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.block(x)
        
        out = out + identity
        
        out = self.act(out)
        return out

class ResBottleNeck(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels,
        GridSize: int = 64,
        norm_method: Literal['batch', 'layer'] = 'batch',
        padding='same',
        padding_mode='reflect',
        act = nn.ReLU()
        ):

        super().__init__()
        mid_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, padding=padding, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, padding=padding, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, padding=padding, padding_mode=padding_mode)
        self.act = act

        match norm_method:
            case 'batch':
                self.bn1 = nn.BatchNorm2d(mid_channels)
                self.bn2 = nn.BatchNorm2d(mid_channels)
                self.bn3 = nn.BatchNorm2d(out_channels)

            case 'layer':
                self.bn1 = nn.LayerNorm([mid_channels, GridSize, GridSize])
                self.bn2 = nn.LayerNorm([mid_channels, GridSize, GridSize])
                self.bn3 = nn.LayerNorm([out_channels, GridSize, GridSize])
            case _:
                self.bn1 = nn.Identity()
                self.bn2 = nn.Identity()
                self.bn3 = nn.Identity()

        self.block = nn.Sequential(
            self.conv1, 
            self.bn1,
            self.act,
            self.conv2,
            self.bn2,
            self.act,
            self.conv3,
            self.bn3,
        )

        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x):
        identity = self.downsample(x)

        out = self.block(x)

        out = out + identity
        
        out = self.act(out)
        return out

class InputHeader(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels,  
        GridSize,
        norm_method: Literal['batch', 'layer'] = 'batch',
        padding='same',
        padding_mode='reflect',
        act = nn.ReLU()
        ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.act = act
        match norm_method:
            case 'batch':
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)

            case 'layer':
                self.bn1 = nn.LayerNorm([out_channels, GridSize, GridSize])
                self.bn2 = nn.LayerNorm([out_channels, GridSize, GridSize])
            case _:
                self.bn1 = nn.Identity()
                self.bn2 = nn.Identity()

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.act,
            self.conv2,
            self.bn2,
            self.act,
        )

    def forward(self, x):
        out = self.net(x)
        return out

class OutputHeader(nn.Module):
    def __init__(
        self,
        in_channels, 
        classes,  
        padding='same',
        padding_mode='reflect',
        end_act=nn.Identity(),
        act=nn.ReLU()):
        super().__init__()
        
        conv1 =  nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=padding,
            padding_mode=padding_mode
        )
        conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=classes,
            kernel_size=1,
            padding=padding,
            padding_mode=padding_mode
        )
        self.net = nn.Sequential(
            conv1, act, conv2, end_act
        ) 
        
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(
        self,
        Block: Type[Union[ResBasicBlock, ResBottleNeck]],
        in_channels: int,
        GridSize: int,
        layer_nums: List[int],
        factor: int,
        norm_method: Literal['batch', 'layer'] = 'batch', 
        pool_method: Literal['max', 'avg'] = 'max',
        padding='same',
        padding_mode='reflect',
        act=nn.ReLU()
        ):
        super().__init__()
        self.in_channels = in_channels
        self.GridSize = GridSize
        self.layers_num = layer_nums
        self.factor = factor
        self.norm_method = norm_method
        self.padding = padding
        self.padding_mode = padding_mode
        self.act = act

        match pool_method:
            case 'max':
                self.pool = nn.MaxPool2d(2, 2)
            case 'avg':
                self.pool = nn.AvgPool2d(2, 2)
            case _:
                self.pool = nn.Identity()

        layers = []
        for num_blocks in layer_nums:
            layers.append(
                self._make_layer(
                    Block, num_blocks, self.in_channels,
                )
            )
            self.in_channels = self.factor * self.in_channels
            self.GridSize = int(self.GridSize / self.factor)
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            x = self.pool(x)
            features.append(x)
        return features
    
    def _make_layer(
        self,
        Block,
        num_blocks,
        in_channels,
        ):
        out_channels = self.factor * in_channels
        blocks = [
            Block(
                in_channels, 
                out_channels, 
                self.GridSize, 
                self.norm_method,
                self.padding,
                self.padding_mode,
                self.act
                )
        ]
        for _ in range(1, num_blocks):
            blocks.append(
                Block(
                    out_channels,
                    out_channels,
                    self.GridSize,
                    self.norm_method,
                    self.padding,
                    self.padding_mode,
                    self.act
                    )
                )
        return nn.Sequential(*blocks)

class Decoder(nn.Module):
    def __init__(
        self,
        Block:Type[Union[ResBasicBlock, ResBottleNeck]],
        out_channels: int,
        GridSize: int,
        layer_nums: List[int],
        factor: int = 2,
        norm_method: Literal['batch', 'layer'] = 'batch',
        padding='same',
        padding_mode='reflect',
        act = nn.ReLU()
        ):
        super().__init__()
        self.out_channels = out_channels
        self.GridSize = GridSize
        self.layers_num = layer_nums
        self.factor = factor
        self.norm_method = norm_method
        self.padding = padding
        self.padding_mode = padding_mode
        self.act = act

        layers = []
        for num_blocks in layer_nums:
            self.GridSize = int(self.GridSize // self.factor)
            layers.append(
                self._make_layer(
                    Block, num_blocks, self.out_channels
                )
            )
            self.out_channels = self.factor * self.out_channels

        layers = layers[::-1]
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, features):
        for feature, layer in zip(features[::-1], self.layers):
            x = x + feature
            x = layer(x)
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        return x

    def _make_layer(
        self,
        Block,
        num_blocks,
        out_channels,
        ):
        in_channels = self.factor * out_channels 
        blocks = [
            Block(
                in_channels,
                out_channels,
                self.GridSize,
                self.norm_method,
                self.padding,
                self.padding_mode,
                self.act
            )
        ]
        for _ in range(num_blocks-1):
            blocks.append(
                Block(
                    out_channels,
                    out_channels,
                    self.GridSize,
                    self.norm_method,
                    self.padding,
                    self.padding_mode,
                    self.act
                    )
                )
        return nn.Sequential(*blocks)



class Sin(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return torch.sin(x)

class UNet(nn.Module):
    def __init__(
        self,
        Block: str,
        planes:int,
        in_channels: int,
        classes,
        GridSize: int,
        layer_nums: List[int],
        factor: int,
        norm_method: Literal['batch', 'layer'],
        pool_method: Literal['max', 'avg'],
        padding='same',
        padding_mode='reflect',
        end_padding='same',
        end_padding_mode = 'reflect',
        center=True,
        act="relu",
        end_act=None,
        ):
        super().__init__()
        match Block:
            case 'ResBottleNeck':
                self.Block = ResBottleNeck
            case 'ResBasic':
                self.Block = ResBasicBlock
            case 'Basic':
                self.Block = BasicBlock
        self.planes = planes
        self.in_channels = in_channels
        self.classes = classes
        self.GridSize = GridSize
        self.layer_nums = layer_nums
        self.factor = factor
        self.norm_method = norm_method
        self.pool_method = pool_method
        self.padding = padding
        self.padding_mode = padding_mode
        self.act_name = act
        match act:
            case 'relu':
               self.act = nn.ReLU()
            case 'tanh':
                self.act = nn.Tanh()
            case 'sin':
                self.act = Sin()
        
        match end_act:
            case 'relu':
               self.end_act = nn.ReLU()
            case 'tanh':
                self.end_act = nn.Tanh()
            case 'sin':
                self.end_act = Sin()
            case _:
                self.end_act = nn.Identity()
                
        self.input_header = InputHeader(
            in_channels=in_channels,
            out_channels=planes,
            GridSize=GridSize,
            norm_method=norm_method,
            padding=padding,
            padding_mode=padding_mode,
            act=self.act
        )

        self.encoder = Encoder(
            self.Block,
            planes, 
            GridSize,
            layer_nums,
            factor, 
            norm_method,
            pool_method,
            padding,
            padding_mode,
            self.act
        )

        if center:
            channels = planes * factor**len(layer_nums)
            self.center = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=padding,
                padding_mode=padding_mode
            )
        else:
            self.center = nn.Identity()

        self.decoder = Decoder(
            self.Block,
            planes, 
            GridSize,
            layer_nums,
            factor, 
            norm_method,
            padding,
            padding_mode,
            self.act
        )
        
        self.output_header = OutputHeader(
            in_channels=planes,
            classes=classes,
            padding=end_padding,
            padding_mode=end_padding_mode,
            act = self.act,
            end_act=self.end_act,
        )
    
    def name(self):
        return "UNet"
    
    def forward(self, x):
        x = self.input_header(x)
        features = self.encoder(x)

        y = self.center(features[-1])
        y = self.decoder(y, features)
        y = self.output_header(y)
        return y
    

class CNN(nn.Module):
    def __init__(self, name, kwargs, dtype=torch.float, device='cpu'):
        super().__init__()
        match name:
            case 'UNet':
                net = UNet(**kwargs)
        self.net = net.to(dtype).to(device)

    def forward(self, x):
        return self.net(x)
    
    @property
    def name(self):
        return self.net.name