import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, List
from typing import Type, Union
import segmentation_models_pytorch as smp

class DoubleConv(nn.Module):
    def __init__(self, inc, outc, mode='reflect'):
        super().__init__()
        layers = [
            nn.Conv2d(inc, outc, 3, 1, padding='same', padding_mode=mode),
            nn.BatchNorm2d(outc),
            nn.ReLU(),
            nn.Conv2d(outc, outc, 3, 1, padding='same', padding_mode=mode),
            nn.BatchNorm2d(outc),
            nn.ReLU()
            ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class UNetEncoder(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        layers = [
            # nn.Conv2d(inc, inc, 5, 2, padding=2, padding_mode='reflect'),
            # nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.5)),
            nn.MaxPool2d(2, 2),
            DoubleConv(inc, outc),
        ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class UNetDecoder(nn.Module):
    def __init__(self, inc, outc, kernel_size=(2, 2), step_size=(2, 2)):
        super().__init__()
        self.decoder = nn.ConvTranspose2d(inc, outc, kernel_size, step_size)
        self.dconv = DoubleConv(outc, outc)
    
    def forward(self, x1, x2):
        x1 = self.decoder(x1)
        x1 = F.interpolate(x1, x2.size()[-2:], align_corners=False, mode='bilinear')
        x = x1 + x2
        # x = torch.concatenate([x1, x2], axis=1)
        return self.dconv(x)

class VaryUNet(nn.Module):
    '''
    Params:
        in_c: input channels.
        out_c: output channels.
        features: The first layers' features.In the following layers,every Encoder will double the features as output channels.
        layers: How deep the UNet is.
        end_padding: The padding mode in the last one conv layer. For finite difference method with point type mesh, end_padding should be 'valid' for Dirichlet type. And (0, 1) for Mixed type boundary.
    '''
    def __init__(self, in_channels=2, classes=1, features=8, layers=5, end_padding=(0, 1)):
        super().__init__()
        self.in_c = in_channels
        self.out_c = classes
        self.features = features
        self.layers = layers

        self.first = nn.Conv2d(in_channels, features, 3, 1, padding='same', padding_mode='reflect')
        self.decoders = []
        self.encoders = []
        for i in range(layers):
            self.encoders.append(
                UNetEncoder(2**i * features, 
                        2**(i+1) * features))
            
            self.decoders.append(
                UNetDecoder(2**(layers-i) * features , 
                        2**(layers-i-1) * features))

        # Middle and End layers
        self.mid = nn.Conv2d(2**(layers) * features, 2**(layers) * features, 1, 1, padding='valid', bias = False)
        self.end = nn.Conv2d(features, classes, 3, 1, padding=end_padding, padding_mode='reflect')
        
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        Ys = [self.first(x)]
        for encoder in self.encoders:
            X = Ys[-1]
            Ys.append(encoder(X))

        Ys.append(self.mid(Ys.pop()))

        for decoder in self.decoders:
            X1 = Ys.pop()
            X2 = Ys.pop()
            Y = decoder(X1, X2)
            
            Ys.append(Y)
        X = Ys.pop()
        return self.end(X)

    @property
    def name(self):
        return f'VaryUNet-{self.in_c}-{self.out_c}-{self.features}-{self.layers}'

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
        # self.relu = nn.ReLU(inplace=True)

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
        padding_mode='reflect',):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=padding, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=padding, padding_mode=padding_mode)
        self.relu = nn.ReLU()

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
            self.relu,
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
        
        out = self.relu(out)
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
        ):

        super().__init__()
        mid_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, padding=padding, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, padding=padding, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, padding=padding, padding_mode=padding_mode)
        self.relu = nn.ReLU()

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
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
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
        
        out = self.relu(out)
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
        self.relu = nn.ReLU()
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
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
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
        padding_mode='reflect'):
        super().__init__()
        
        self.conv =  nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=padding,
            padding_mode=padding_mode
        )
        self.act = nn.ReLU()

        self.header = nn.Conv2d(
            in_channels=in_channels,
            out_channels=classes,
            kernel_size=1,
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        out = self.header(x)
        return out

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
        ):
        super().__init__()
        self.in_channels = in_channels
        self.GridSize = GridSize
        self.layers_num = layer_nums
        self.factor = factor
        self.norm_method = norm_method
        self.padding = padding
        self.padding_mode = padding_mode

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
                    Block, num_blocks, self.in_channels
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
        ):
        super().__init__()
        self.out_channels = out_channels
        self.GridSize = GridSize
        self.layers_num = layer_nums
        self.factor = factor
        self.norm_method = norm_method
        self.padding = padding
        self.padding_mode = padding_mode

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
                self.padding_mode
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
                    self.padding_mode
                    )
                )
        return nn.Sequential(*blocks)


class Adaptor(nn.Module):
    '''
    Adaptor is a optimizable component which connects the encoder and decoder on the same level.
    It can be any forms. 
    For example, it could be a Basic Residual block or a Attention block and so on. 
    '''
    def __init__(
        self,
        Block:Type[Union[ResBasicBlock, ResBottleNeck]],
        in_channels: int,
        out_channels: int,
        GridSize: int,
        num_blocks: int,
        norm_method: Literal['batch', 'layer'] = 'batch', 
        padding='same',
        padding_mode='reflect',
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels=out_channels
        self.GridSize = GridSize
        self.norm_method = norm_method
        self.padding = padding
        self.padding_mode = padding_mode
        self.layers = self._make_layer(Block, num_blocks, in_channels, out_channels)
        
    def forward(self, x):
        return self.layers(x)

    def _make_layer(
        self,
        Block,
        num_blocks,
        in_channels,
        out_channels,
        ):
        layers = [
            Block(
                    in_channels,
                    out_channels,
                    self.GridSize,
                    self.norm_method,
                    self.padding,
                    self.padding_mode
                    )
                ]

        for _ in range(num_blocks-1):
            layers.append(
                Block(
                    out_channels,
                    out_channels,
                    self.GridSize,
                    self.norm_method,
                    self.padding,
                    self.padding_mode
                )
            )
        return nn.Sequential(*layers)

class SegModel(nn.Module):
    def __init__(
        self,
        Block: str,
        planes:int,
        in_channels: int,
        classes,
        GridSize: int,
        layer_nums: List[int],
        adaptor_nums:List[int],
        factor: int,
        norm_method: Literal['batch', 'layer'],
        pool_method: Literal['max', 'avg'],
        padding='same',
        end_padding='same',
        padding_mode='reflect',
        end_padding_mode = 'reflect',
        center=True,
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
        self.adaptor_nums = adaptor_nums
        self.factor = factor
        self.norm_method = norm_method
        self.pool_method = pool_method
        self.padding = padding
        self.padding_mode = padding_mode

        self.input_header = InputHeader(
            in_channels=in_channels,
            out_channels=planes,
            GridSize=GridSize,
            norm_method=norm_method,
            padding=padding,
            padding_mode=padding_mode
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
        
        self.adaptors = self._make_adaptors()

        self.decoder = Decoder(
            self.Block,
            planes, 
            GridSize,
            layer_nums,
            factor, 
            norm_method,
            padding,
            padding_mode ,
        )
        
        self.output_header = OutputHeader(
            in_channels=planes,
            classes=classes,
            padding=end_padding,
            padding_mode=end_padding_mode
        )
    
    @property
    def name(self):
        layer_name = "#".join([str(num) for num in self.layer_nums])
        adaptor_name = "#".join([str(num) for num in  self.adaptor_nums])
        return f"{self.Block.__name__}-{layer_name}-{adaptor_name}-{self.factor}-{self.norm_method}-{self.pool_method}-{self.padding_mode}"
    
    def _make_adaptors(self):
        adaptors = []
        for i, num_blocks in enumerate(self.adaptor_nums):
            channels = self.planes * self.factor**(i+1)
            GridSize = int(self.GridSize // ((self.factor)**(i+1)))
            adaptors.append(
                Adaptor(
                    self.Block,
                    in_channels=channels,
                    out_channels=channels,
                    GridSize=GridSize,
                    num_blocks=num_blocks,
                    norm_method=self.norm_method,
                    padding=self.padding,
                    padding_mode=self.padding_mode,
                )
            )
        adaptors = nn.ModuleList(adaptors)
        return adaptors
    
    def forward(self, x):
        x = self.input_header(x)
        features = self.encoder(x)
        adapted_features = []

        for feature, adaptor in zip(features, self.adaptors):
            adapted_features.append(
                adaptor(feature)
            )
        y = self.center(features[-1])
        y = self.decoder(y, adapted_features)
        y = self.output_header(y)
        return y
    
class CNN(nn.Module):
    def __init__(self, name, kwargs, dtype=torch.float, device='cpu'):
        super().__init__()
        match name:
            case 'segmodel':
                net = SegModel(**kwargs)
            
            case 'varyunet':
                net = VaryUNet(**kwargs)

            case 'unet++':
                net = smp.UnetPlusPlus(**kwargs)
            
            case 'unet':
                net = smp.Unet(**kwargs)
        self.net = net.to(dtype).to(device)

    def forward(self, x):
        return self.net(x)
    
    @property
    def name(self):
        return self.net.name