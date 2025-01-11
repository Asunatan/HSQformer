import math

import torch
from matplotlib import pyplot as plt
from timm.layers import to_2tuple, trunc_normal_
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange,repeat
from .convnextv2_segmentaion import ConvNeXtV2
from .convnext_segmentaion import ConvNeXt
from timm.models.layers import  DropPath
from .swin_transformer_segmentation import SwinTransformer
from .cswin_transformer_segmentation import CSWin
from .hiera_segmentation import Hiera
#from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .ops_dcnv3.modules.dcnv3 import DCNv3

#from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # elif self.data_format == "channels_first":
        #     u = x.mean(1, keepdim=True)
        #     s = (x - u).pow(2).mean(1, keepdim=True)
        #     x = (x - u) / torch.sqrt(s + self.eps)
        #     x = self.weight[:, None, None] * x + self.bias[:, None, None]
        #     return x
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            std = x.std(1, keepdim=True)
            return self.weight[:, None, None] * (x - mean) / (std + self.eps) + self.bias[:, None, None]
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.dwconv_small = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)+self.dwconv_small(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
class liver_cov_net(nn.Module):
    def __init__(self,num_classes=2,deep_supervision=False,drop_path_rate=0.2,**kwargs):
        super(liver_cov_net, self).__init__()
        self.tumor_backbone = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],drop_path_rate=0.1, **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1, **kwargs)
        self.original_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=0.2, **kwargs)
        self.num_classes=num_classes
        self.tumor_backbone = self.load_pretrained(self.tumor_backbone, 'tiny')
        self.liver_backbone = self.load_pretrained(self.liver_backbone, 'small')
        self.original_backbone = self.load_pretrained(self.original_backbone, 'base')
        depths = [3, 3, 27, 3]
        fuse_depths = [ 3, 9, 3]
        fuse_dims = [320, 640, 1280, 2560]
        dims=[128, 256, 512, 1024]
        stage_fuse_dims = [ 512, 1024,2048]
        self.fuse_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in range(4):
            stem = nn.Sequential(
                nn.Conv2d(fuse_dims[i], dims[i], kernel_size=3, stride=1,padding=1),
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.GELU()
            )
            self.fuse_layers.append(stem)

        self.fuse_stagelayers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in range(3):
            stem = nn.Sequential(
                nn.Conv2d(stage_fuse_dims[i], dims[i+1], kernel_size=3, stride=1,padding=1),
                LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
                nn.GELU()
            )
            self.fuse_stagelayers.append(stem)

        #stage_Block
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        layer_scale_init_value = 1e-6
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        # fuse_stage_Block
        fuse_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(fuse_depths))]
        cur = 0
        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i+1], drop_path=fuse_dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(fuse_depths[i])]
            )
            self.fuse_stages.append(stage)
            cur += fuse_depths[i]


        self.downsample_layers= nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        # self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
        #                             nn.BatchNorm2d(48),
        #                             nn.ReLU(True))
        #
        # self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True),
        #                           nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def load_pretrained(self,backbone,type):
        if type == 'tiny':
            weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_tiny_22k_224.pth')['model']
        elif type=='small':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth')['model']
        elif type=='base':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_base_22k_224.pth')['model']
        model_dict = {}
        state_dict = backbone.state_dict()
        for k, v in weight.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        backbone.load_state_dict(state_dict)
        inchannel = backbone.head.in_features
        backbone.head = nn.Linear(in_features=inchannel, out_features=self.num_classes)
        return backbone

    # def forward(self, x1,x2,x3,aux_loss=False):
    #     x1 = self.original_backbone(x1)#dims=[128, 256, 512, 1024]
    #     # x2 = self.liver_backbone(x2)#dims=[96, 192, 384, 768]
    #     # x3 = self.tumor_backbone(x3)#dims=[96, 192, 384, 768]
    #     # x=[]#dims=[128, 256, 512, 1024]
    #     # for i in range(4):
    #     #     out=self.fuse_layers[i](torch.cat((x1[i],x2[i],x3[i]),dim=1))
    #     #     out=self.stages[i](out)
    #     #     x.append(out)
    #     #
    #     # fuse_feature = x[0]
    #     # for i in range(3):
    #     #     down = self.downsample_layers[i](fuse_feature)
    #     #     fuse = self.fuse_stagelayers[i](torch.cat((down, x[i + 1]), dim=1))
    #     #     fuse = self.fuse_stages[i](fuse)
    #     #     fuse_feature = fuse
    #
    #     # result=self.norm(fuse_feature.mean([-2, -1]))
    #     # result=self.classifier(result)
    #
    #     result=self.norm(x1[-1].mean([-2, -1]))
    #     result=self.classifier(result)
    #
    #     return result

    def forward(self, x1,x2,x3,aux_loss=False):#original_img,liver_img,tumor_img
        if aux_loss:
            x1,aux_original = self.original_backbone(x1,aux_loss=True)  # dims=[128, 256, 512, 1024]
            x2,aux_liver = self.liver_backbone(x2,aux_loss=True)  # dims=[96, 192, 384, 768]
            x3,aux_tumor = self.tumor_backbone(x3,aux_loss=True)  # dims=[96, 192, 384, 768]
        else:
            x1 = self.original_backbone(x1)#dims=[128, 256, 512, 1024]
            x2 = self.liver_backbone(x2)#dims=[96, 192, 384, 768]
            x3 = self.tumor_backbone(x3)#dims=[96, 192, 384, 768]
        x=[]#dims=[128, 256, 512, 1024]
        for i in range(4):
            out=self.fuse_layers[i](torch.cat((x1[i],x2[i],x3[i]),dim=1))
            out=self.stages[i](out)
            x.append(out)

        fuse_feature = x[0]
        for i in range(3):
            down = self.downsample_layers[i](fuse_feature)
            fuse = self.fuse_stagelayers[i](torch.cat((down, x[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            fuse_feature = fuse

        result=self.norm(fuse_feature.mean([-2, -1]))
        result=self.classifier(result)
        if aux_loss:
            return [result,aux_original,aux_liver,aux_tumor]
        return result

        # if aux_loss:
        #     aux=self.norm(x[-1].mean([-2, -1]))
        #     aux=self.head(aux)
        #     return x,aux
        # else:
        #     return x
class liver_cswin_net(nn.Module):
    def __init__(self,num_classes=2,deep_supervision=False,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_cswin_net, self).__init__()
        self.aux_loss=aux_loss
        self.tumor_backbone = CSWin(embed_dim=64, depth=[1,2,21,1],drop_path_rate=0.3,aux_loss=aux_loss,
        split_size=[1,2,7,7], num_heads=[2,4,8,16], **kwargs)
        self.liver_backbone = CSWin(embed_dim=64, depth=[2,4,32,2],drop_path_rate=0.3,aux_loss=aux_loss,
        split_size=[1,2,7,7], num_heads=[2,4,8,16],**kwargs)
        self.original_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=0.3,aux_loss=aux_loss,
        split_size=[1,2,7,7], num_heads=[4,8,16,32], **kwargs)
        self.num_classes=num_classes
        self.tumor_backbone = self.load_pretrained(self.tumor_backbone, 'tiny')
        self.liver_backbone = self.load_pretrained(self.liver_backbone, 'small')
        self.original_backbone = self.load_pretrained(self.original_backbone, 'base')
        depths = [3, 3, 3, 3]
        fuse_depths = [ 3, 3, 3]
        fuse_dims = [224, 448, 896, 1792]#第一次聚合的维度
        dims=[128, 256, 512, 1024]
        stage_fuse_dims = [ 512, 1024,2048]#第二次聚合的维度
        self.fuse_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in range(4):
            stem = nn.Sequential(
                nn.Conv2d(fuse_dims[i], dims[i], kernel_size=3, stride=1,padding=1),
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.GELU()
            )
            self.fuse_layers.append(stem)

        self.fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(3):
            stem = nn.Sequential(
                nn.Conv2d(stage_fuse_dims[i], dims[i+1], kernel_size=3, stride=1,padding=1),
                LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
                nn.GELU()
            )
            self.fuse_stagelayers.append(stem)

        #stage_Block  #第一次三个拼起来后用stage_Block聚合
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        layer_scale_init_value = 1e-6
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        # fuse_stage_Block
        fuse_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(fuse_depths))]
        cur = 0
        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i+1], drop_path=fuse_dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(fuse_depths[i])]
            )
            self.fuse_stages.append(stage)
            cur += fuse_depths[i]


        self.downsample_layers= nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        # self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
        #                             nn.BatchNorm2d(48),
        #                             nn.ReLU(True))
        #
        # self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True),
        #                           nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def load_pretrained(self,backbone,type):
        if type == 'tiny':
            weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_tiny_224.pth')['state_dict_ema']
        elif type=='small':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_small_224.pth')['state_dict_ema']
        elif type=='base':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')['state_dict_ema']
        model_dict = {}
        state_dict = backbone.state_dict()
        for k, v in weight.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        backbone.load_state_dict(state_dict)
        inchannel = backbone.head.in_features
        backbone.head = nn.Linear(in_features=inchannel, out_features=self.num_classes)
        return backbone

    # def forward(self, x1,x2,x3,aux_loss=False):
    #     x1 = self.original_backbone(x1)#dims=[128, 256, 512, 1024]
    #     # x2 = self.liver_backbone(x2)#dims=[96, 192, 384, 768]
    #     # x3 = self.tumor_backbone(x3)#dims=[96, 192, 384, 768]
    #     # x=[]#dims=[128, 256, 512, 1024]
    #     # for i in range(4):
    #     #     out=self.fuse_layers[i](torch.cat((x1[i],x2[i],x3[i]),dim=1))
    #     #     out=self.stages[i](out)
    #     #     x.append(out)
    #     #
    #     # fuse_feature = x[0]
    #     # for i in range(3):
    #     #     down = self.downsample_layers[i](fuse_feature)
    #     #     fuse = self.fuse_stagelayers[i](torch.cat((down, x[i + 1]), dim=1))
    #     #     fuse = self.fuse_stages[i](fuse)
    #     #     fuse_feature = fuse
    #
    #     # result=self.norm(fuse_feature.mean([-2, -1]))
    #     # result=self.classifier(result)
    #
    #     result=self.norm(x1[-1].mean([-2, -1]))
    #     result=self.classifier(result)
    #
    #     return result

    def forward(self, x1,x2,x3):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x1,aux_original = self.original_backbone(x1)  # dims=[128, 256, 512, 1024]
            x2,aux_liver = self.liver_backbone(x2)  # dims=[96, 192, 384, 768]
            x3,aux_tumor = self.tumor_backbone(x3)  # dims=[96, 192, 384, 768]
        else:
            x1 = self.original_backbone(x1)#dims=[96, 192, 384, 768]
            x2 = self.liver_backbone(x2)#dims=[64, 128, 256, 512]
            x3 = self.tumor_backbone(x3)#dims=[64,128,256,512],
        x=[]#dims=[128, 256, 512, 1024]
        for i in range(4):

            out=self.fuse_layers[i](torch.cat((x1[i],x2[i],x3[i]),dim=1))
            out=self.stages[i](out)
            x.append(out)

        fuse_feature = x[0]
        for i in range(3):
            down = self.downsample_layers[i](fuse_feature)
            fuse = self.fuse_stagelayers[i](torch.cat((down, x[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            fuse_feature = fuse

        result=self.norm(fuse_feature.mean([-2, -1]))
        result=self.classifier(result)
        if self.aux_loss:
            return [result,aux_original,aux_liver,aux_tumor]
        return result

        # if aux_loss:
        #     aux=self.norm(x[-1].mean([-2, -1]))
        #     aux=self.head(aux)
        #     return x,aux
        # else:
        #     return x
class liver_conv_cswin_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.1,**kwargs):
        super(liver_conv_cswin_net, self).__init__()
        self.aux_loss=aux_loss
        self.tumor_backbone = CSWin(embed_dim=64, depth=[1,2,21,1],aux_loss=aux_loss,drop_path_rate=0.3,
        split_size=[1,2,7,7], num_heads=[2,4,8,16], **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=0.3,aux_loss=aux_loss, **kwargs)
        self.original_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=0.3,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss, **kwargs)
        self.num_classes=num_classes
        self.tumor_backbone = self.load_pretrained(self.tumor_backbone, 'tiny')
        self.liver_backbone = self.load_pretrained(self.liver_backbone, 'small')
        self.original_backbone = self.load_pretrained(self.original_backbone, 'base')
        depths = [3, 3, 9, 3]
        fuse_depths = [ 3, 9, 3]
        fuse_dims = [288, 576, 1152, 2304]#第一次聚合的维度
        dims=[128, 256, 512, 1024]
        stage_fuse_dims = [ 512, 1024,2048]#第二次聚合的维度
        self.fuse_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in range(4):
            stem = nn.Sequential(
                nn.Conv2d(fuse_dims[i], dims[i], kernel_size=1, stride=1,padding=0),
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.GELU()
            )
            self.fuse_layers.append(stem)

        self.fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(3):
            stem = nn.Sequential(
                nn.Conv2d(stage_fuse_dims[i], dims[i+1], kernel_size=1, stride=1,padding=0),
                LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
                nn.GELU()
            )
            self.fuse_stagelayers.append(stem)

        #stage_Block  #第一次三个拼起来后用stage_Block聚合
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        layer_scale_init_value = 1e-6
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        # fuse_stage_Block
        fuse_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(fuse_depths))]
        cur = 0
        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i+1], drop_path=fuse_dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(fuse_depths[i])]
            )
            self.fuse_stages.append(stage)
            cur += fuse_depths[i]


        self.downsample_layers= nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2,padding=1),
            )
            self.downsample_layers.append(downsample_layer)
        # self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
        #                             nn.BatchNorm2d(48),
        #                             nn.ReLU(True))
        #
        # self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True),
        #                           nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def load_pretrained(self,backbone,type):
        if type == 'tiny':
            weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_tiny_224.pth')['state_dict_ema']
        elif type=='small':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_base_22k_224.pth')['model']
        elif type=='base':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')['state_dict_ema']
        model_dict = {}
        state_dict = backbone.state_dict()
        for k, v in weight.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        backbone.load_state_dict(state_dict)
        inchannel = backbone.head.in_features
        backbone.head = nn.Linear(in_features=inchannel, out_features=self.num_classes)
        return backbone

    def forward(self, x1,x2,x3):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x1,aux_original = self.original_backbone(x1)  # dims=[128, 256, 512, 1024]
            x2,aux_liver = self.liver_backbone(x2)  # dims=[96, 192, 384, 768]
            x3,aux_tumor = self.tumor_backbone(x3)  # dims=[96, 192, 384, 768]
        else:
            x1 = self.original_backbone(x1)#dims=[96, 192, 384, 768]
            x2 = self.liver_backbone(x2)#dims=[128, 256, 512, 1024]
            x3 = self.tumor_backbone(x3)#dims=[64,128,256,512],
        x=[]#dims=[128, 256, 512, 1024]
        for i in range(4):

            out=self.fuse_layers[i](torch.cat((x1[i],x2[i],x3[i]),dim=1))
            out=self.stages[i](out)
            x.append(out)

        fuse_feature = x[0]
        for i in range(3):
            down = self.downsample_layers[i](fuse_feature)
            fuse = self.fuse_stagelayers[i](torch.cat((down, x[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            fuse_feature = fuse

        result=self.norm(fuse_feature.mean([-2, -1]))
        result=self.classifier(result)
        if self.aux_loss:
            return [result,aux_original,aux_liver,aux_tumor]
        return result

        # if aux_loss:
        #     aux=self.norm(x[-1].mean([-2, -1]))
        #     aux=self.head(aux)
        #     return x,aux
        # else:
        #     return x
class liver_conv2cswin_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.,**kwargs):
        super(liver_conv2cswin_net, self).__init__()
        self.aux_loss=aux_loss
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=0.3,aux_loss=aux_loss, **kwargs)
        self.original_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=0.3,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss, **kwargs)
        self.num_classes=num_classes
        self.liver_backbone = self.load_pretrained(self.liver_backbone, 'ConvNeXt')
        self.original_backbone = self.load_pretrained(self.original_backbone, 'CSWin')
        depths = [1, 1, 3, 1]
        fuse_depths = [ 1, 3, 1]
        fuse_dims = [224, 448, 896, 1792]#第一次聚合的维度
        dims=[128, 256, 512, 1024]

        stage_fuse_dims = [ 512, 1024,2048]#第二次聚合的维度
        self.fuse_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in range(4):
            stem = nn.Sequential(
                nn.Conv2d(fuse_dims[i], dims[i], kernel_size=3, stride=1,padding=1),
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                #nn.GELU()
            )
            self.fuse_layers.append(stem)

        #stage_Block  #第一次三个拼起来后用stage_Block聚合
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        layer_scale_init_value = 1e-6
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(3):
            stem = nn.Sequential(
                nn.Conv2d(stage_fuse_dims[i], dims[i+1], kernel_size=3, stride=1,padding=1),
                LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
                #nn.GELU()
            )
            self.fuse_stagelayers.append(stem)

        # fuse_stage_Block
        fuse_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(fuse_depths))]
        cur = 0
        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i+1], drop_path=fuse_dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(fuse_depths[i])]
            )
            self.fuse_stages.append(stage)
            cur += fuse_depths[i]


        self.downsample_layers= nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
        #                             nn.BatchNorm2d(48),
        #                             nn.ReLU(True))
        #
        # self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True),
        #                           nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def load_pretrained(self,backbone,type):
        if type == 'ConvNeXt':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_base_22k_224.pth')['model']
        elif type=='CSWin':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')['state_dict_ema']
        else:
            print('err')
            exit()
        model_dict = {}
        state_dict = backbone.state_dict()
        for k, v in weight.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        backbone.load_state_dict(state_dict)
        inchannel = backbone.head.in_features
        backbone.head = nn.Linear(in_features=inchannel, out_features=self.num_classes)
        return backbone


    def forward(self, x1,x2):#original_img,liver_img
        if self.aux_loss:
            x1,aux_original = self.original_backbone(x1)  # dims=[128, 256, 512, 1024]
            x2,aux_liver = self.liver_backbone(x2)  # dims=[96, 192, 384, 768]

        else:
            x1 = self.original_backbone(x1)#dims=[96, 192, 384, 768]
            x2 = self.liver_backbone(x2)#dims=[128, 256, 512, 1024]
        x=[]#dims=[128, 256, 512, 1024]
        for i in range(4):

            out=self.fuse_layers[i](torch.cat((x1[i],x2[i]),dim=1))
            out=self.stages[i](out)
            x.append(out)

        fuse_feature = x[0]
        for i in range(3):
            down = self.downsample_layers[i](fuse_feature)
            fuse = self.fuse_stagelayers[i](torch.cat((down, x[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            fuse_feature = fuse

        result=self.norm(fuse_feature.mean([-2, -1]))
        result=self.classifier(result)
        if self.aux_loss:
            return [result,aux_original,aux_liver]
        return result

        # if aux_loss:
        #     aux=self.norm(x[-1].mean([-2, -1]))
        #     aux=self.head(aux)
        #     return x,aux
        # else:
        #     return x
class liver1_detr_net(nn.Module):
    def __init__(self,num_classes=2,deep_supervision=False,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver1_detr_net, self).__init__()
        self.aux_loss = aux_loss
        self.num_classes=num_classes
        self.liver_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=0.3,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss, **kwargs)
        self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
                             transformer_dim_feedforward=1024,transformer_enc_layers=6,)
        #self.predictor = MultiScaleMaskedTransformerDecoder(in_channels=256,num_queries=2, num_classes=num_classes)
        self.liver_backbone = self.load_pretrained(self.liver_backbone, 'CSWin')
        # inchannel = self.liver_backbone.head.in_features
        # self.liver_backbone.head = nn.Linear(in_features=inchannel, out_features=self.num_classes)
        depths = [1, 1, 3, 1]
        fuse_depths = [1, 3, 1]
        fuse_dims = [224, 448, 896, 1792]  # 第一次聚合的维度
        dims = [256, 512, 768, 1024]

        stage_fuse_dims = [768, 1024, 1280]  # 第二次聚合的维度
        self.fuse_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        for i in range(4):
            stem = nn.Sequential(
                nn.Conv2d(fuse_dims[i], dims[i], kernel_size=3, stride=1, padding=1),
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.GELU()
            )
            self.fuse_layers.append(stem)

        # stage_Block  #第一次三个拼起来后用stage_Block聚合
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        layer_scale_init_value = 1e-6
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.fuse_stagelayers = nn.ModuleList()  # 第一次三个拼起来后用stage_Block聚合
        for i in range(3):
            stem = nn.Sequential(
                nn.Conv2d(stage_fuse_dims[i], dims[i + 1], kernel_size=3, stride=1, padding=1),
                LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first"),
                nn.GELU()
            )
            self.fuse_stagelayers.append(stem)

        # fuse_stage_Block
        fuse_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(fuse_depths))]
        cur = 0
        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i + 1], drop_path=fuse_dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(fuse_depths[i])]
            )
            self.fuse_stages.append(stage)
            cur += fuse_depths[i]

        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        # self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
        #                             nn.BatchNorm2d(48),
        #                             nn.ReLU(True))
        #
        # self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True),
        #                           nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def load_pretrained(self,backbone,type):
        if type == 'ConvNeXt':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_base_22k_224.pth')['model']
        elif type=='CSWin':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')['state_dict_ema']
        else:
            print('err')
            exit()
        model_dict = {}
        state_dict = backbone.state_dict()
        for k, v in weight.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        backbone.load_state_dict(state_dict)
        inchannel = backbone.head.in_features
        backbone.head = nn.Linear(in_features=inchannel, out_features=self.num_classes)
        return backbone
    def forward(self, x,mask=None):
        if self.aux_loss:
            x,aux_result=self.liver_backbone(x)
        else:
            x = self.liver_backbone(x)
        x = self.pixel_decoder.forward_features(x)
        fuse_feature = x[0]
        for i in range(3):
            down = self.downsample_layers[i](fuse_feature)
            fuse = self.fuse_stagelayers[i](torch.cat((down, x[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            fuse_feature = fuse

        result=self.norm(fuse_feature.mean([-2, -1]))
        result=self.classifier(result)
        if self.aux_loss:
            return [result,aux_result]
        return result

        #predictions = self.predictor(multi_scale_features, mask_features, mask)
        #return predictions
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        #context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        #context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask = None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask = None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask = None,
                query_pos= None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead=8, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        #self.multihead_attn =CrossAttention(d_model,d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos= None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,memory_mask= None,memory_key_padding_mask= None,
                    pos = None,query_pos= None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory, memory_mask = None, memory_key_padding_mask = None,
                pos = None,query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation =  nn.GELU()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class PatchEmbed(nn.Module):

    def __init__(self, patch_size=7,img_size=224,in_chans=3, out_channel=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        #self.proj = nn.Conv2d(in_chans, out_channel, kernel_size=patch_size, stride=patch_size,)
        self.proj = nn.Conv2d(in_chans, out_channel, kernel_size=3, stride=1,padding=1)
        self.norm = nn.LayerNorm(out_channel)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, out_channel))
        #self.proj_linear=nn.Linear(out_channel,out_channel)
        #self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(-1, -2)#+self.position_embeddings
        #x=self.proj_linear(x)
        #x = self.dropout(self.norm(x))
        #x=self.norm(x)
        return x

class Cross_3_AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048,dropout=0.1,depths = [1, 1, 3, 1], normalize_before=False):
        super().__init__()
        self.self_attention_q = SelfAttentionLayer(d_model=d_model, nhead=nhead, dropout=dropout,normalize_before=normalize_before)
        self.cross_attention_q = CrossAttentionLayer(d_model=d_model, nhead=nhead, dropout=dropout, normalize_before=normalize_before)
        self.self_fuse_attention_q = SelfAttentionLayer(d_model=d_model, nhead=nhead, dropout=dropout, normalize_before=normalize_before)
        #self.local_feature = Block(dim=d_model)
        self.ffn_q = FFNLayer(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, normalize_before=normalize_before)

        self.self_attention_kv = SelfAttentionLayer(d_model=d_model, nhead=nhead, dropout=dropout,normalize_before=normalize_before)
        self.cross_attention_kv = CrossAttentionLayer(d_model=d_model, nhead=nhead, dropout=dropout, normalize_before=normalize_before)
        self.self_fuse_attention_kv = SelfAttentionLayer(d_model=d_model, nhead=nhead, dropout=dropout, normalize_before=normalize_before)
        #self.local_feature0 = Block(dim=d_model)
        self.ffn_kv = FFNLayer(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, normalize_before=normalize_before)
    def forward(self, q,kv):
        # xkv=kv
        #B, N, C = kv.shape
        #xkv_local = kv.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
        xkv = self.cross_attention_kv(q, kv)+self.self_attention_kv(kv)#+self.local_feature(xkv_local).flatten(2).transpose(1, 2)
        xkv = self.self_fuse_attention_kv(xkv)

        xkv = self.ffn_kv(xkv)

        # xq=q
        #xq_local = q.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
        xq = self.cross_attention_q(kv, q)+self.self_attention_q(q)#+self.local_feature0(xq_local).flatten(2).transpose(1, 2)
        xq = self.self_fuse_attention_q(xq)

        xq = self.ffn_q(xq)

        return xq,xkv

class Cross_2classification_AttentionBlock(nn.Module):
    def __init__(self, q_dim,kv_dim, nhead=8, dim_feedforward=2048,dropout=0.0,depths = [1, 1, 3, 1], normalize_before=True):
        super().__init__()
        self.self_attention_q = SelfAttentionLayer(d_model=q_dim, nhead=nhead, dropout=dropout,normalize_before=normalize_before)
        self.cross_attention_q = CrossAttentionLayer(d_model=q_dim, nhead=nhead, dropout=dropout, normalize_before=normalize_before)
        self.self_fuse_attention_q = SelfAttentionLayer(d_model=q_dim, nhead=nhead, dropout=dropout, normalize_before=normalize_before)
        #self.local_feature = Block(dim=d_model)
        self.ffn_q = FFNLayer(d_model=q_dim, dim_feedforward=2*q_dim, dropout=dropout, normalize_before=normalize_before)

        self.self_attention_kv = SelfAttentionLayer(d_model=kv_dim, nhead=nhead, dropout=dropout,normalize_before=normalize_before)
        self.cross_attention_kv = CrossAttentionLayer(d_model=kv_dim, nhead=nhead, dropout=dropout, normalize_before=normalize_before)
        self.self_fuse_attention_kv = SelfAttentionLayer(d_model=kv_dim, nhead=nhead, dropout=dropout, normalize_before=normalize_before)
        #self.local_feature0 = Block(dim=d_model)
        self.ffn_kv = FFNLayer(d_model=kv_dim, dim_feedforward=2*kv_dim, dropout=dropout, normalize_before=normalize_before)
    def forward(self, q,kv):
        # xkv=kv
        #B, N, C = kv.shape
        #xkv_local = kv.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
        xkv = self.cross_attention_kv(q, kv)+self.self_attention_kv(kv)#+self.local_feature(xkv_local).flatten(2).transpose(1, 2)
        xkv = self.self_fuse_attention_kv(xkv)
        xkv = self.ffn_kv(xkv)

        # xq=q
        #xq_local = q.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
        xq = self.cross_attention_q(kv, q)+self.self_attention_q(q)#+self.local_feature0(xq_local).flatten(2).transpose(1, 2)
        xq = self.self_fuse_attention_q(xq)
        xq = self.ffn_q(xq)

        return xq,xkv

class liver3_cswin_liner_cross_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.1,**kwargs):
        super(liver3_cswin_liner_cross_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=0.3,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=True, **kwargs)
        #self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=0.3,aux_loss=aux_loss, **kwargs)
        self.liver_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=0.2,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=True,  **kwargs)
        self.original_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=0.1,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=True,  **kwargs)
        self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
                             transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        self.tumor_backbone = self.load_pretrained(self.tumor_backbone, 'tiny')
        self.liver_backbone = self.load_pretrained(self.liver_backbone, 'small')
        self.original_backbone = self.load_pretrained(self.original_backbone, 'base')
        self.cross_attention_depths=[1,3,1]
        self.cross_attention_dim = [192,384,768]
        self.cross_fuse_dims = [384, 768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [576, 1152, 2304]  # 每一个阶段concat
        self.frist_depths = [2]
        self.frist_dims = [96]
        #deformed之后的特征金子塔
        self.depths = [2, 2, 9, 2]
        self.fuse_depths = [2, 9, 2]
        self.fuse_dims = [288, 576, 1152, 2304]#第一次聚合的维度
        self.dims=[256, 512, 1024,2048]
        self.stage_fuse_dims = [ 768, 1280,2304]#第二次聚合的维度



        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(len(self.cross_attention_depths)):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i], nhead=8, dropout=0.0,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)

        self.cross_fuse_tumor_liver = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(3):
            stem = nn.Sequential(
                nn.Linear(self.cross_fuse_dims[i], self.cross_attention_dim[i]),
                nn.LayerNorm(self.cross_attention_dim[i])
                # nn.Conv2d(cross_fuse_dims[i], cross_attention_dim[i], kernel_size=1, stride=1,padding=0),
                # nn.BatchNorm2d(dims[i]),
                #LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                #nn.GELU()
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(3):
            stem = nn.Sequential(
                nn.Linear(self.fuse_stage_dims[i], self.cross_attention_dim[i]),
                nn.LayerNorm(self.cross_attention_dim[i]),
                nn.GELU()
            )
            self.cross_fuse_stagelayers.append(stem)

        #stage_Block  #用于将三个基分类器的第一个阶段进行融合
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        layer_scale_init_value = 1e-6
        self.frist_tumor_stages= nn.Sequential(
                *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
            )
        self.frist_liver_stages= nn.Sequential(
                *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
            )
        self.frist_original_stages= nn.Sequential(
                *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
            )
        self.frist_fuse_stages= nn.Sequential(
                nn.Conv2d(3*self.frist_dims[0], self.frist_dims[0], kernel_size=1, stride=1,padding=0),
                # nn.BatchNorm2d(288),
                Block(dim=self.frist_dims[0]),Block(dim=self.frist_dims[0]),
                #LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
                #nn.GELU()
            )
        # self.frist_tumor_stages = nn.ModuleList()
        # self.frist_liver_stages = nn.ModuleList()
        # self.frist_original_stages = nn.ModuleList()
        # for i in range(1):
        #     stage = nn.Sequential(
        #         *[Block(dim=self.frist_dims[i], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[i])]
        #     )
        #     self.frist_tumor_stages.append(stage)
        #     self.frist_liver_stages.append(stage)
        #     self.frist_original_stages.append(stage)
        #     cur += self.frist_depths[i]
        # fuse_stage_Block

        self.downsample_layers= nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, stride=2,padding=1),
            )
            self.downsample_layers.append(downsample_layer)
        self.fuse_stagelayers= nn.ModuleList()
        for i in range(3):
            layer = nn.Sequential(
                    nn.Conv2d(self.stage_fuse_dims[i], self.dims[i+1], kernel_size=3, stride=1,padding=1),

            )
            self.fuse_stagelayers.append(layer)

        fuse_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.fuse_depths))]
        cur = 0
        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=self.dims[i+1], drop_path=fuse_dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(self.fuse_depths[i])]
            )
            self.fuse_stages.append(stage)
            cur += self.fuse_depths[i]



        self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)
        self.classifier = nn.Linear(self.dims[-1], num_classes)
        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
    def forward(self, x1,x2,x3):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x1)  # dims=[128, 256, 512, 1024]
            x_liver,aux_liver = self.liver_backbone(x2)  # dims=[96, 192, 384, 768]
            x_tumor,aux_tumor = self.tumor_backbone(x3)  # dims=[96, 192, 384, 768]
        else:
            x_original = self.original_backbone(x1)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x2)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x3)#dims=[64,128,256,512],
        # for i in range(len(self.cross_tumor_liver[0])):
        #     out1, out2 = self.cross_tumor_liver[0][i](out1, out2)
        x = []
        x_tumor[0]=self.frist_tumor_stages(x_tumor[0])
        x_liver[0] = self.frist_liver_stages(x_liver[0])
        x_original[0] = self.frist_original_stages(x_original[0])
        x.append(self.frist_fuse_stages(torch.cat((x_tumor[0],x_liver[0],x_original[0]),dim=1)))



        cross_out = []
        for i in range(3):
            out1, out2 = x_tumor[i + 1], x_liver[i + 1]
            for layer in self.cross_tumor_liver[i]:
                out1, out2 = layer(out1, out2)
            cross_out.append(self.cross_fuse_tumor_liver[i](torch.cat((out1, out2), dim=2)))
            out1, out2 = x_tumor[i + 1], x_original[i + 1]
            for layer in self.cross_tumor_original[i]:
                out1, out2 = layer(out1, out2)
            cross_out.append(self.cross_fuse_tumor_original[i](torch.cat((out1, out2), dim=2)))
            out1, out2 = x_liver[i + 1], x_original[i + 1]
            for layer in self.cross_liver_original[i]:
                out1, out2 = layer(out1, out2)
            cross_out.append(self.cross_fuse_liver_original[i](torch.cat((out1, out2), dim=2)))
            x.append(self.cross_fuse_stagelayers[i](torch.cat(cross_out, dim=2)))
            cross_out = []

        for i in range(1,4):
            B,N,C = x[i].shape
            x[i] = x[i].transpose(1, 2).view(B,-1 ,int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
        x = self.pixel_decoder.forward_features(x)
        # x = []
        #
        # cross_out=[]#dims=[128, 256, 512, 1024]
        # out1, out2 = x_tumor[1], x_liver[1]
        # for layer in self.cross_tumor_liver[0]:
        #     out1, out2 = layer(out1, out2)
        # cross_out.append(self.cross_fuse_tumor_liver[0](torch.cat((out1,out2),dim=2)))
        # out1, out2 = x_tumor[1], x_original[1]
        # for layer in self.cross_tumor_original[0]:
        #     out1, out2 = layer(out1, out2)
        # cross_out.append(self.cross_fuse_tumor_original[0](torch.cat((out1,out2),dim=2)))
        # out1, out2 = x_liver[1], x_original[1]
        # for layer in self.cross_liver_original[0]:
        #     out1, out2 = layer(out1, out2)
        # cross_out.append(self.cross_fuse_liver_original[0](torch.cat((out1,out2),dim=2)))
        # x.append(self.fuse_stagelayers[0](torch.cat(cross_out,dim=2)))
        #
        # cross_out = []
        # out1, out2 = x_tumor[2], x_liver[2]
        # for layer in self.cross_tumor_liver[1]:
        #     out1, out2 = layer(out1, out2)
        # cross_out.append(self.cross_fuse_tumor_liver[1](torch.cat((out1,out2),dim=2)))
        # out1, out2 = x_tumor[2], x_original[2]
        # for layer in self.cross_tumor_original[1]:
        #     out1, out2 = layer(out1, out2)
        # cross_out.append(self.cross_fuse_tumor_original[1](torch.cat((out1,out2),dim=2)))
        # out1, out2 = x_liver[2], x_original[2]
        # for layer in self.cross_liver_original[1]:
        #     out1, out2 = layer(out1, out2)
        # cross_out.append(self.cross_fuse_liver_original[1](torch.cat((out1,out2),dim=2)))
        # x.append(self.fuse_stagelayers[1](torch.cat(cross_out,dim=2)))
        #
        # cross_out = []
        # out1, out2 = x_tumor[3], x_liver[3]
        # for layer in self.cross_tumor_liver[2]:
        #     out1, out2 = layer(out1, out2)
        # cross_out.append(self.cross_fuse_tumor_liver[2](torch.cat((out1,out2),dim=2)))
        # out1, out2 = x_tumor[3], x_original[3]
        # for layer in self.cross_tumor_original[2]:
        #     out1, out2 = layer(out1, out2)
        # cross_out.append(self.cross_fuse_tumor_original[2](torch.cat((out1,out2),dim=2)))
        # out1, out2 = x_liver[3], x_original[3]
        # for layer in self.cross_liver_original[2]:
        #     out1, out2 = layer(out1, out2)
        # cross_out.append(self.cross_fuse_liver_original[2](torch.cat((out1,out2),dim=2)))
        # x.append(self.fuse_stagelayers[2](torch.cat(cross_out,dim=2)))
        fuse_feature = x[0]
        for i in range(3):
            down = self.downsample_layers[i](fuse_feature)
            fuse = self.fuse_stagelayers[i](torch.cat((down, x[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            fuse_feature = fuse

        result=self.norm(fuse_feature.mean([-2, -1]))
        result=self.classifier(result)

        if self.aux_loss:
            return [result,aux_original,aux_liver,aux_tumor],self.vote_weights
        return result

    def load_pretrained(self,backbone,type):
        if type == 'tiny':
            weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')['state_dict_ema']
        elif type=='small':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')['state_dict_ema']
        elif type=='base':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')['state_dict_ema']
        model_dict = {}
        state_dict = backbone.state_dict()
        for k, v in weight.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        backbone.load_state_dict(state_dict)
        inchannel = backbone.head.in_features
        backbone.head = nn.Linear(in_features=inchannel, out_features=self.num_classes)
        return backbone
class liver3_cswin_cross_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.0,**kwargs):
        super(liver3_cswin_cross_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=0.1,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=True, **kwargs)
        #self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=0.3,aux_loss=aux_loss, **kwargs)
        self.liver_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=0.2,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=True,  **kwargs)
        self.original_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=0.3,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=True,  **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        self.tumor_backbone = self.load_pretrained(self.tumor_backbone, 'tiny')
        self.liver_backbone = self.load_pretrained(self.liver_backbone, 'small')
        self.original_backbone = self.load_pretrained(self.original_backbone, 'base')
        self.cross_attention_depths=[1,3,1]
        self.cross_attention_dim = [192,384,768]
        self.cross_fuse_dims = [384, 768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [576, 1152, 2304]  # 交叉注意力每一个阶段concat
        #self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.frist_depths = [2]
        self.frist_dims = [96]
        #deformed之后的特征金子塔
        self.depths = [2]
        self.fuse_depths = [2, 2, 2]
        self.dims=[96, 192, 384,768]
        self.stage_fuse_dims = [ 384, 768,1536]#第二次聚合的维度



        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(len(self.cross_attention_depths)):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i], nhead=8, dropout=0.0,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)

        self.cross_fuse_tumor_liver = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(3):
            stem = nn.Sequential(
                # nn.Linear(self.cross_fuse_dims[i], self.cross_attention_dim[i]),
                # nn.LayerNorm(self.cross_attention_dim[i])
                nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1,padding=0),
                nn.BatchNorm2d(self.cross_attention_dim[i]),
                nn.GELU()
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(3):
            stem = nn.Sequential(
                nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),
                #LayerNorm(self.cross_attention_dim[i], eps=1e-6),
                nn.BatchNorm2d(self.cross_attention_dim[i]),
                nn.GELU()
            )
            self.cross_fuse_stagelayers.append(stem)

        #stage_Block  #用于将三个基分类器的第一个阶段进行融合
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        layer_scale_init_value = 1e-6
        self.frist_tumor_stages= nn.Sequential(
                *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
            )
        self.frist_liver_stages= nn.Sequential(
                *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
            )
        self.frist_original_stages= nn.Sequential(
                *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
            )
        self.frist_fuse_stages= nn.Sequential(
                nn.Conv2d(3*self.frist_dims[0], self.frist_dims[0], kernel_size=1, stride=1,padding=0),
                Block(dim=self.frist_dims[0]),Block(dim=self.frist_dims[0]),
                nn.BatchNorm2d(self.frist_dims[0]),
                nn.GELU()
            )

        self.downsample_layers= nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(self.dims[i]),
                nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, stride=2,padding=1),
            )
            self.downsample_layers.append(downsample_layer)
        self.fuse_stagelayers= nn.ModuleList()
        for i in range(3):
            layer = nn.Sequential(
                nn.Conv2d(self.stage_fuse_dims[i], self.dims[i+1], kernel_size=3, stride=1,padding=1),
                # nn.BatchNorm2d(self.dims[i+1]),
                # nn.GELU()
            )
            self.fuse_stagelayers.append(layer)

        fuse_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.fuse_depths))]
        cur = 0
        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=self.dims[i+1], drop_path=fuse_dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(self.fuse_depths[i])]
            )
            self.fuse_stages.append(stage)
            cur += self.fuse_depths[i]



        self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6) # final norm layer
        self.classifier = nn.Linear(self.dims[-1], num_classes)
        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        #self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x1,x2,x3):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x1)  # dims=[128, 256, 512, 1024]
            x_liver,aux_liver = self.liver_backbone(x2)  # dims=[96, 192, 384, 768]
            x_tumor,aux_tumor = self.tumor_backbone(x3)  # dims=[96, 192, 384, 768]
            # 检查每个层的输出是否有NAN
            # for i, output in enumerate([x_original,x_liver,x_tumor,aux_original,aux_liver,aux_tumor]):
            #     if torch.isnan(output).any():
            #         print(f'第{i}层输出出现了NAN!')
            #         break
        else:
            x_original = self.original_backbone(x1)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x2)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x3)#dims=[64,128,256,512],
        # for i in range(len(self.cross_tumor_liver[0])):
        #     out1, out2 = self.cross_tumor_liver[0][i](out1, out2)
        x = []
        x_tumor[0]=self.frist_tumor_stages(x_tumor[0])
        x_liver[0] = self.frist_liver_stages(x_liver[0])
        x_original[0] = self.frist_original_stages(x_original[0])
        x.append(self.frist_fuse_stages(torch.cat((x_tumor[0],x_liver[0],x_original[0]),dim=1)))

        cross_out = []
        for i in range(3):
            B,N,C = x_tumor[i + 1].shape
            out1, out2 = x_tumor[i + 1], x_liver[i + 1]

            for layer in self.cross_tumor_liver[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
            cross_out.append(self.cross_fuse_tumor_liver[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_tumor[i + 1], x_original[i + 1]
            for layer in self.cross_tumor_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
            cross_out.append(self.cross_fuse_tumor_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_liver[i + 1], x_original[i + 1]
            for layer in self.cross_liver_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
            cross_out.append(self.cross_fuse_liver_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            x.append(self.cross_fuse_stagelayers[i](torch.cat(cross_out, dim=1)))
            cross_out = []

        # for i in range(1,4):
        #     B,N,C = x[i].shape
        #     x[i] = x[i].transpose(1, 2).view(B,C ,int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
        #x = self.pixel_decoder.forward_features(x)
        fuse_feature = x[0]
        for i in range(3):
            down = self.downsample_layers[i](fuse_feature)
            fuse = self.fuse_stagelayers[i](torch.cat((down, x[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            fuse_feature = fuse

        result=self.norm(fuse_feature.mean([-2, -1]))
        result=self.classifier(result)
        # for i, output in enumerate([result,aux_original,aux_liver,aux_tumor]):
        #     if torch.isnan(output).any():
        #         print(f'第{i}层输出出现了NAN!')
        #         break
        if self.aux_loss:
            return [result,aux_original,aux_liver,aux_tumor]#,self.vote_weights
        return result

    def load_pretrained(self,backbone,type):
        if type == 'tiny':
            weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')['state_dict_ema']
        elif type=='small':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')['state_dict_ema']
        elif type=='base':
            weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth')['state_dict_ema']
        model_dict = {}
        state_dict = backbone.state_dict()
        for k, v in weight.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        backbone.load_state_dict(state_dict)
        inchannel = backbone.head.in_features
        backbone.head = nn.Linear(in_features=inchannel, out_features=self.num_classes)
        return backbone


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()#build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class InternImageLayer(nn.Module):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.1,
                 drop_path=0.1,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=True,
                 layer_scale=1.0,
                 offset_scale=1.0,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 =nn.LayerNorm(channels,eps=1e-6) #build_norm_layer(channels, 'LN')e-
        self.post_norm = post_norm
        self.dcn = core_op(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale) # for InternImage-H/G
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = nn.LayerNorm(channels,eps=1e-6) #build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = nn.LayerNorm(channels,eps=1e-6)#build_norm_layer(channels, 'LN')
            self.res_post_norm2 = nn.LayerNorm(channels,eps=1e-6)#build_norm_layer(channels, 'LN')

    def forward(self, x):
        x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x
        # if self.post_norm:
        #     x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
        #     x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
        # else:
        #     x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
        #     x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        # return x


class liver_3_feature_extract(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_feature_extract,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=drop_path_rate,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=False, **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3),drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        #三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]

        self.cross_attention_depths=[3,3]
        self.cross_attention_dim = [384,768] #交叉注意力维度
        self.cross_fuse_dims = [768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        #self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288,576]
        #deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims=[96, 192, 384,768]
        self.stage_fuse_dims = [ 384, 768,1536]#第二次聚合的维度

        # stage_Block  #用于将三个基分类器的第一个阶段进行融合
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # cur = 0
        # layer_scale_init_value = 1e-6
        # 考虑是否必要
        # self.frist_tumor_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        # self.frist_liver_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        # self.frist_original_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        self.frist_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.frist_dcn = nn.Sequential(
            #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[0] // 3,groups=self.fuse_dcn_dims[0] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )

        self.second_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.second_dcn = nn.Sequential(
            #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[1] // 3,groups=self.fuse_dcn_dims[1] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        # x_original_dims=[96, 192, 384, 768]
        # x_liver_dims=[96, 192, 384, 768]
        # x_tumor_dims=[96, 192, 384, 768]
        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(len(self.cross_attention_depths)):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i],dim_feedforward=4*self.cross_attention_dim[i], nhead=8, dropout=0.1,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)


        self.cross_fuse_tumor_liver = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(2):
            stem = nn.Sequential(
                # nn.Linear(self.cross_fuse_dims[i], self.cross_attention_dim[i]),
                # nn.LayerNorm(self.cross_attention_dim[i])
                nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1,padding=0),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU(),
                # Block()
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(2):
            stem = nn.Sequential(
                nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),

                #LayerNorm(self.cross_attention_dim[i], eps=1e-6),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU()
            )
            self.cross_fuse_stagelayers.append(stem)


        self.downsample_layers= nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, stride=2,padding=1),
            )
            self.downsample_layers.append(downsample_layer)
        self.fuse_stagelayers= nn.ModuleList()
        for i in range(3):
            layer = nn.Sequential(
                nn.Conv2d(self.stage_fuse_dims[i], self.dims[i+1], kernel_size=3, stride=1,padding=1),
                # nn.BatchNorm2d(self.dims[i+1]),
                # nn.GELU()
            )
            self.fuse_stagelayers.append(layer)

        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(*[Block(dim=self.dims[i+1],drop_path=0.1) for j in range(self.fuse_depths[i])]
            )
            self.fuse_stages.append(stage)

        self.classifier_weight = nn.Parameter(torch.full(size=(8,), fill_value=0.125),requires_grad=True)
        self.classifier = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6),nn.Linear(self.dims[0], num_classes))
        self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6),nn.Linear(self.dims[1], num_classes))
        self.head2 = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6),nn.Linear(self.dims[2], num_classes))
        self.head3 = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        #self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]
            x_liver,aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor,aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]

        else:
            x_original = self.original_backbone(x)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)#dims=[64,128,256,512],
        # for i in range(len(self.cross_tumor_liver[0])):
        #     out1, out2 = self.cross_tumor_liver[0][i](out1, out2)
        feature = []
        #这一步似乎没有必要,直接拼起来聚合特征就好，待考虑
        # x_tumor[0]=self.frist_tumor_stages(x_tumor[0])
        # x_liver[0] = self.frist_liver_stages(x_liver[0])
        # x_original[0] = self.frist_original_stages(x_original[0])
        x=self.frist_fuse(torch.cat((x_tumor[0],x_liver[0],x_original[0]),dim=1)).permute(0,2,3,1)
        x=self.frist_dcn(x).permute(0,3,1,2)
        feature.append(x)

        x=self.second_fuse(torch.cat((x_tumor[1],x_liver[1],x_original[1]),dim=1)).permute(0,2,3,1)
        x=self.second_dcn(x).permute(0,3,1,2)
        feature.append(x)
        #.transpose(-1, -2)

        for i in range(2):
            cross_out = []
            B,C,H,W = x_tumor[i+2].shape
            x_tumor[i+2]= x_tumor[i+2].flatten(2).transpose(-1, -2)
            x_liver[i + 2]=x_liver[i+2].flatten(2).transpose(-1, -2)
            x_original[i + 2]=x_original[i + 2].flatten(2).transpose(-1, -2)
            out1, out2 = x_tumor[i + 2], x_liver[i + 2]
            for layer in self.cross_tumor_liver[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_liver[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_tumor[i + 2], x_original[i + 2]
            for layer in self.cross_tumor_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_liver[i + 2], x_original[i + 2]
            for layer in self.cross_liver_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_liver_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            feature.append(self.cross_fuse_stagelayers[i](torch.cat(cross_out, dim=1)))
            #cross_out = []

        # for i in range(1,4):
        #     B,N,C = x[i].shape
        #     x[i] = x[i].transpose(1, 2).view(B,C ,int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
        #x = self.pixel_decoder.forward_features(x)
        x = feature[0]
        for i in range(3):
            down = self.downsample_layers[i](x)
            fuse = self.fuse_stagelayers[i](torch.cat((down, feature[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            x = fuse


        mulit_scale_head = self.classifier(x.mean([-2, -1]))


        if self.aux_loss:
            head0 = self.head0(feature[0].mean([-2, -1]))
            head1 = self.head1(feature[1].mean([-2, -1]))
            head2 = self.head2(feature[2].mean([-2, -1]))
            head3 = self.head3(feature[3].mean([-2, -1]))
            # print(feature[0].mean([-2, -1]).shape)
            #weighted_pred = torch.zeros_like(result)
            weighted_pred = 0.
            for i ,pre in enumerate([mulit_scale_head,head0,head1,head2,head3,aux_original,aux_liver,aux_tumor]):
                weighted_pred += self.classifier_weight[i]*pre
                #weighted_pred.append(self.classifier_weight[i]*pre)

            return x.mean([-2, -1]),weighted_pred

            #return [result,head0,head1,head2,head3,aux_original,aux_liver,aux_tumor]#,self.vote_weights
        else:
            return None

    def load_pretrained(self):
        tumor_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth',map_location="cpu")['state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth',map_location="cpu")['model_state']
        # self.tumor_backbone.head.out_features = self.num_classes
        # self.liver_backbone.head.out_features = self.num_classes
        # self.original_backbone.head.projection.out_features = self.num_classes
        self.tumor_backbone.load_state_dict(tumor_weight,strict=False)
        self.liver_backbone.load_state_dict(liver_weight,strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features, out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(in_features=self.original_backbone.head.projection.in_features,
                                             out_features=self.num_classes)
        del tumor_weight,liver_weight,original_weight

class liver_3_classification_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_classification_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=drop_path_rate,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=False, **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3),drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        #三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]

        self.cross_attention_depths=[3,3]
        self.cross_attention_dim = [384,768] #交叉注意力维度
        self.cross_fuse_dims = [768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        #self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288,576]
        #deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims=[96, 192, 384,768]
        self.stage_fuse_dims = [ 384, 768,1536]#第二次聚合的维度

        # stage_Block  #用于将三个基分类器的第一个阶段进行融合
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # cur = 0
        # layer_scale_init_value = 1e-6
        # 考虑是否必要
        # self.frist_tumor_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        # self.frist_liver_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        # self.frist_original_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        self.frist_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.frist_dcn = nn.Sequential(
            #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[0] // 3,groups=self.fuse_dcn_dims[0] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )

        self.second_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.second_dcn = nn.Sequential(
            #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[1] // 3,groups=self.fuse_dcn_dims[1] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        # x_original_dims=[96, 192, 384, 768]
        # x_liver_dims=[96, 192, 384, 768]
        # x_tumor_dims=[96, 192, 384, 768]
        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(len(self.cross_attention_depths)):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i],dim_feedforward=4*self.cross_attention_dim[i], nhead=8, dropout=0.1,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)


        self.cross_fuse_tumor_liver = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(2):
            stem = nn.Sequential(
                # nn.Linear(self.cross_fuse_dims[i], self.cross_attention_dim[i]),
                # nn.LayerNorm(self.cross_attention_dim[i])
                nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1,padding=0),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU(),
                # Block()
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(2):
            stem = nn.Sequential(
                nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),

                #LayerNorm(self.cross_attention_dim[i], eps=1e-6),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU()
            )
            self.cross_fuse_stagelayers.append(stem)


        self.downsample_layers= nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, stride=2,padding=1),
            )
            self.downsample_layers.append(downsample_layer)
        self.fuse_stagelayers= nn.ModuleList()
        for i in range(3):
            layer = nn.Sequential(
                nn.Conv2d(self.stage_fuse_dims[i], self.dims[i+1], kernel_size=3, stride=1,padding=1),
                # nn.BatchNorm2d(self.dims[i+1]),
                # nn.GELU()
            )
            self.fuse_stagelayers.append(layer)

        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(*[Block(dim=self.dims[i+1],drop_path=0.1) for j in range(self.fuse_depths[i])]
            )
            self.fuse_stages.append(stage)

        self.classifier_weight = nn.Parameter(torch.full(size=(8,), fill_value=0.125),requires_grad=True)
        self.classifier = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6),nn.Linear(self.dims[0], num_classes))
        self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6),nn.Linear(self.dims[1], num_classes))
        self.head2 = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6),nn.Linear(self.dims[2], num_classes))
        self.head3 = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        #self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):#original_img,liver_img,tumor_img
        # visual_feature=[]
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]
            x_liver,aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor,aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]
            # visual_feature.append((x_original,x_liver,x_tumor))
            # B, C, H, W = x_original[2].shape
            # for i in range(B):
            #     image = x_original[2][i].detach().cpu().numpy().transpose(1, 2, 0)
            #     if True:
            #         image = image.sum(-1)
            #         fig = plt.figure()
            #         ax = fig.add_subplot(1, 1, 1)
            #         ax.imshow(image)
            #         plt.axis('off')
            #         plt.savefig(f'/home/uax/SCY/LiverClassification/feature_map/{i}_{0}.png')
            #         plt.close()
            #     else:
            #         for c in range(C):
            #             feature_map_c=image[:,:,c]
            #             fig = plt.figure()
            #             ax = fig.add_subplot(1,1,1)
            #             ax.imshow(feature_map_c)
            #             plt.axis('off')
            #             plt.savefig(f'/home/uax/SCY/LiverClassification/feature_map/{i}_{c}.png')
            #             plt.close()
            # a=0
        else:
            x_original = self.original_backbone(x)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)#dims=[64,128,256,512],
        # for i in range(len(self.cross_tumor_liver[0])):
        #     out1, out2 = self.cross_tumor_liver[0][i](out1, out2)
        feature = []
        #这一步似乎没有必要,直接拼起来聚合特征就好，待考虑
        # x_tumor[0]=self.frist_tumor_stages(x_tumor[0])
        # x_liver[0] = self.frist_liver_stages(x_liver[0])
        # x_original[0] = self.frist_original_stages(x_original[0])
        x=self.frist_fuse(torch.cat((x_tumor[0],x_liver[0],x_original[0]),dim=1)).permute(0,2,3,1)
        x=self.frist_dcn(x).permute(0,3,1,2)
        feature.append(x)

        x=self.second_fuse(torch.cat((x_tumor[1],x_liver[1],x_original[1]),dim=1)).permute(0,2,3,1)
        x=self.second_dcn(x).permute(0,3,1,2)
        feature.append(x)
        #.transpose(-1, -2)

        for i in range(2):
            cross_out = []
            B,C,H,W = x_tumor[i+2].shape
            x_tumor[i+2]= x_tumor[i+2].flatten(2).transpose(-1, -2)
            x_liver[i + 2]=x_liver[i+2].flatten(2).transpose(-1, -2)
            x_original[i + 2]=x_original[i + 2].flatten(2).transpose(-1, -2)
            out1, out2 = x_tumor[i + 2], x_liver[i + 2]
            for layer in self.cross_tumor_liver[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_liver[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_tumor[i + 2], x_original[i + 2]
            for layer in self.cross_tumor_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_liver[i + 2], x_original[i + 2]
            for layer in self.cross_liver_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_liver_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            feature.append(self.cross_fuse_stagelayers[i](torch.cat(cross_out, dim=1)))
            #cross_out = []

        # for i in range(1,4):
        #     B,N,C = x[i].shape
        #     x[i] = x[i].transpose(1, 2).view(B,C ,int(math.sqrt(N)), int(math.sqrt(N))).contiguous()
        #x = self.pixel_decoder.forward_features(x)
        x = feature[0]
        for i in range(3):
            down = self.downsample_layers[i](x)
            fuse = self.fuse_stagelayers[i](torch.cat((down, feature[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            x = fuse


        mulit_scale_head = self.classifier(x.mean([-2, -1]))


        if self.aux_loss:
            head0 = self.head0(feature[0].mean([-2, -1]))
            head1 = self.head1(feature[1].mean([-2, -1]))
            head2 = self.head2(feature[2].mean([-2, -1]))
            head3 = self.head3(feature[3].mean([-2, -1]))
            #weighted_pred = torch.zeros_like(result)
            weighted_pred = []
            for i ,pre in enumerate([mulit_scale_head,head0,head1,head2,head3,aux_original,aux_liver,aux_tumor]):
                #weighted_pred += self.classifier_weight[i]*pre
                weighted_pred.append(self.classifier_weight[i]*pre)
            return weighted_pred

            #return [result,head0,head1,head2,head3,aux_original,aux_liver,aux_tumor]#,self.vote_weights
        else:
            return None

    def load_pretrained(self):
        tumor_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth',map_location="cpu")['state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth',map_location="cpu")['model_state']
        # self.tumor_backbone.head.out_features = self.num_classes
        # self.liver_backbone.head.out_features = self.num_classes
        # self.original_backbone.head.projection.out_features = self.num_classes
        self.tumor_backbone.load_state_dict(tumor_weight,strict=False)
        self.liver_backbone.load_state_dict(liver_weight,strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features, out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(in_features=self.original_backbone.head.projection.in_features,
                                             out_features=self.num_classes)
        del tumor_weight,liver_weight,original_weight
class liver_3_ablation_nostage0_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_ablation_nostage0_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=drop_path_rate,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=False, **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3),drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        #三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]

        self.cross_attention_depths=[3,3]
        self.cross_attention_dim = [384,768] #交叉注意力维度
        self.cross_fuse_dims = [768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        #self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288,576]
        #deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims=[96, 192, 384,768]
        self.stage_fuse_dims = [ 384, 768,1536]#第二次聚合的维度

        # self.frist_fuse = nn.Sequential(
        #     nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
        #     # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
        #     # nn.BatchNorm2d(self.frist_dims//3),
        #     # nn.GELU()
        # )
        # self.frist_dcn = nn.Sequential(
        #     #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
        #     InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[0] // 3,groups=self.fuse_dcn_dims[0] // 3),
        #     InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
        #     # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
        #     # nn.BatchNorm2d(self.frist_dims//3),
        #     # nn.GELU()
        # )

        self.second_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.second_dcn = nn.Sequential(
            #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[1] // 3,groups=self.fuse_dcn_dims[1] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        # x_original_dims=[96, 192, 384, 768]
        # x_liver_dims=[96, 192, 384, 768]
        # x_tumor_dims=[96, 192, 384, 768]
        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(len(self.cross_attention_depths)):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i],dim_feedforward=4*self.cross_attention_dim[i], nhead=8, dropout=0.1,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)


        self.cross_fuse_tumor_liver = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(2):
            stem = nn.Sequential(
                # nn.Linear(self.cross_fuse_dims[i], self.cross_attention_dim[i]),
                # nn.LayerNorm(self.cross_attention_dim[i])
                nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1,padding=0),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU(),
                # Block()
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(2):
            stem = nn.Sequential(
                nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),

                #LayerNorm(self.cross_attention_dim[i], eps=1e-6),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU()
            )
            self.cross_fuse_stagelayers.append(stem)


        self.downsample_layers= nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, stride=2,padding=1),
            )
            self.downsample_layers.append(downsample_layer)
        self.fuse_stagelayers= nn.ModuleList()
        for i in range(3):
            layer = nn.Sequential(
                nn.Conv2d(self.stage_fuse_dims[i], self.dims[i+1], kernel_size=3, stride=1,padding=1),

            )
            self.fuse_stagelayers.append(layer)

        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(*[Block(dim=self.dims[i+1],) for j in range(self.fuse_depths[i])]
            )
            self.fuse_stages.append(stage)

        self.classifier_weight = nn.Parameter(torch.ones(8),requires_grad=True)
        self.classifier = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6),nn.Linear(self.dims[0], num_classes))
        self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6),nn.Linear(self.dims[1], num_classes))
        self.head2 = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6),nn.Linear(self.dims[2], num_classes))
        self.head3 = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        #self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]
            x_liver,aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor,aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]

        else:
            x_original = self.original_backbone(x)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)#dims=[64,128,256,512],

        feature = []
        # x=self.frist_fuse(torch.cat((x_tumor[0],x_liver[0],x_original[0]),dim=1)).permute(0,2,3,1)
        # x=self.frist_dcn(x).permute(0,3,1,2)
        # feature.append(x)
        feature.append('x')

        x=self.second_fuse(torch.cat((x_tumor[1],x_liver[1],x_original[1]),dim=1)).permute(0,2,3,1)
        x=self.second_dcn(x).permute(0,3,1,2)
        feature.append(x)


        for i in range(2):
            cross_out = []
            B,C,H,W = x_tumor[i+2].shape
            x_tumor[i+2]= x_tumor[i+2].flatten(2).transpose(-1, -2)
            x_liver[i + 2]=x_liver[i+2].flatten(2).transpose(-1, -2)
            x_original[i + 2]=x_original[i + 2].flatten(2).transpose(-1, -2)
            out1, out2 = x_tumor[i + 2], x_liver[i + 2]
            for layer in self.cross_tumor_liver[i]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_liver[i](torch.cat((out1, out2), dim=1)))

            out1, out2 = x_tumor[i + 2], x_original[i + 2]
            for layer in self.cross_tumor_original[i]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_original[i](torch.cat((out1, out2), dim=1)))

            out1, out2 = x_liver[i + 2], x_original[i + 2]
            for layer in self.cross_liver_original[i]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_liver_original[i](torch.cat((out1, out2), dim=1)))

            feature.append(self.cross_fuse_stagelayers[i](torch.cat(cross_out, dim=1)))

        x = feature[1]
        for i in range(1,3):
            down = self.downsample_layers[i](x)
            fuse = self.fuse_stagelayers[i](torch.cat((down, feature[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            x = fuse


        result = self.classifier(x.mean([-2, -1]))


        if self.aux_loss:
            #head0 = self.head0(feature[0].mean([-2, -1]))
            head1 = self.head1(feature[1].mean([-2, -1]))
            head2 = self.head2(feature[2].mean([-2, -1]))
            head3 = self.head3(feature[3].mean([-2, -1]))

            weighted_pred = []
            for i ,pre in enumerate([result,head1,head2,head3,aux_original,aux_liver,aux_tumor]):

                weighted_pred.append(self.classifier_weight[i]*pre)
            return weighted_pred

        else:
            return result

    def load_pretrained(self):
        tumor_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth',map_location="cpu")['state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth',map_location="cpu")['model_state']

        self.tumor_backbone.load_state_dict(tumor_weight,strict=False)
        self.liver_backbone.load_state_dict(liver_weight,strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features, out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(in_features=self.original_backbone.head.projection.in_features,
                                             out_features=self.num_classes)
        del tumor_weight,liver_weight,original_weight
class liver_3_ablation_nostage1_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_ablation_nostage1_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=drop_path_rate,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=False, **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3),drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        #三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]

        self.cross_attention_depths=[3,3]
        self.cross_attention_dim = [384,768] #交叉注意力维度
        self.cross_fuse_dims = [768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        #self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288,576]
        #deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims=[96, 192, 384,768]
        self.stage_fuse_dims = [ 384, 768,1536]#第二次聚合的维度

        self.frist_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.frist_dcn = nn.Sequential(
            #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[0] // 3,groups=self.fuse_dcn_dims[0] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )

        # self.second_fuse = nn.Sequential(
        #     nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
        #     # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
        #     # nn.BatchNorm2d(self.frist_dims//3),
        #     # nn.GELU()
        # )
        # self.second_dcn = nn.Sequential(
        #     #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
        #     InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[1] // 3,groups=self.fuse_dcn_dims[1] // 3),
        #     InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
        #     # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
        #     # nn.BatchNorm2d(self.frist_dims//3),
        #     # nn.GELU()
        # )
        # x_original_dims=[96, 192, 384, 768]
        # x_liver_dims=[96, 192, 384, 768]
        # x_tumor_dims=[96, 192, 384, 768]
        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(len(self.cross_attention_depths)):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i],dim_feedforward=4*self.cross_attention_dim[i], nhead=8, dropout=0.1,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)


        self.cross_fuse_tumor_liver = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(2):
            stem = nn.Sequential(
                # nn.Linear(self.cross_fuse_dims[i], self.cross_attention_dim[i]),
                # nn.LayerNorm(self.cross_attention_dim[i])
                nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1,padding=0),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU(),
                # Block()
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(2):
            stem = nn.Sequential(
                nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),

                #LayerNorm(self.cross_attention_dim[i], eps=1e-6),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU()
            )
            self.cross_fuse_stagelayers.append(stem)


        self.downsample_layers= nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(self.dims[0], self.dims[1], kernel_size=3, stride=2,padding=1),
            nn.Conv2d(self.dims[1], self.dims[2], kernel_size=3, stride=2, padding=1),
            ))
        self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(self.dims[2], self.dims[3], kernel_size=3, stride=2,padding=1),
            ))
        # for i in range(2):
        #     downsample_layer = nn.Sequential(
        #         nn.Conv2d(self.dims[i+1], self.dims[i+2], kernel_size=3, stride=2,padding=1),
        #     )
        #     self.downsample_layers.append(downsample_layer)
        self.fuse_stagelayers= nn.ModuleList()
        for i in range(2):
            layer = nn.Sequential(
                nn.Conv2d(self.stage_fuse_dims[i+1], self.dims[i+2], kernel_size=3, stride=1,padding=1),

            )
            self.fuse_stagelayers.append(layer)

        self.fuse_stages = nn.ModuleList()
        for i in range(2):
            stage = nn.Sequential(*[Block(dim=self.dims[i+2],) for j in range(self.fuse_depths[i])]
            )
            self.fuse_stages.append(stage)

        self.classifier_weight = nn.Parameter(torch.ones(7),requires_grad=True)
        self.classifier = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6),nn.Linear(self.dims[0], num_classes))
        # self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6),nn.Linear(self.dims[1], num_classes))
        self.head2 = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6),nn.Linear(self.dims[2], num_classes))
        self.head3 = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        #self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]
            x_liver,aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor,aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]

        else:
            x_original = self.original_backbone(x)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)#dims=[64,128,256,512],

        feature = []
        x=self.frist_fuse(torch.cat((x_tumor[0],x_liver[0],x_original[0]),dim=1)).permute(0,2,3,1)
        x=self.frist_dcn(x).permute(0,3,1,2)
        feature.append(x)


        # x=self.second_fuse(torch.cat((x_tumor[1],x_liver[1],x_original[1]),dim=1)).permute(0,2,3,1)
        # x=self.second_dcn(x).permute(0,3,1,2)
        # feature.append(x)
        feature.append('X')

        for i in range(2):
            cross_out = []
            B,C,H,W = x_tumor[i+2].shape
            x_tumor[i+2]= x_tumor[i+2].flatten(2).transpose(-1, -2)
            x_liver[i + 2]=x_liver[i+2].flatten(2).transpose(-1, -2)
            x_original[i + 2]=x_original[i + 2].flatten(2).transpose(-1, -2)
            out1, out2 = x_tumor[i + 2], x_liver[i + 2]
            for layer in self.cross_tumor_liver[i]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_liver[i](torch.cat((out1, out2), dim=1)))

            out1, out2 = x_tumor[i + 2], x_original[i + 2]
            for layer in self.cross_tumor_original[i]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_original[i](torch.cat((out1, out2), dim=1)))

            out1, out2 = x_liver[i + 2], x_original[i + 2]
            for layer in self.cross_liver_original[i]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_liver_original[i](torch.cat((out1, out2), dim=1)))

            feature.append(self.cross_fuse_stagelayers[i](torch.cat(cross_out, dim=1)))

        x = feature[0]
        for i in range(2):
            down = self.downsample_layers[i](x)
            fuse = self.fuse_stagelayers[i](torch.cat((down, feature[i + 2]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            x = fuse


        result = self.classifier(x.mean([-2, -1]))


        if self.aux_loss:
            head0 = self.head0(feature[0].mean([-2, -1]))
            #head1 = self.head1(feature[1].mean([-2, -1]))
            head2 = self.head2(feature[2].mean([-2, -1]))
            head3 = self.head3(feature[3].mean([-2, -1]))

            weighted_pred = []
            for i ,pre in enumerate([result,head0,head2,head3,aux_original,aux_liver,aux_tumor]):

                weighted_pred.append(self.classifier_weight[i]*pre)
            return weighted_pred

        else:
            return result

    def load_pretrained(self):
        tumor_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth',map_location="cpu")['state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth',map_location="cpu")['model_state']

        self.tumor_backbone.load_state_dict(tumor_weight,strict=False)
        self.liver_backbone.load_state_dict(liver_weight,strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features, out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(in_features=self.original_backbone.head.projection.in_features,
                                             out_features=self.num_classes)
        del tumor_weight,liver_weight,original_weight
class liver_3_ablation_nostage2_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_ablation_nostage2_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2, 4, 32, 2], drop_path_rate=drop_path_rate,
                                    split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32], aux_loss=aux_loss, cross=False,
                                    **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,
                                       aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), drop_path_rate=drop_path_rate,
                                       aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        # 三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]

        self.cross_attention_depths = [3, 3]
        self.cross_attention_dim = [384, 768]  # 交叉注意力维度
        self.cross_fuse_dims = [768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        # self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288, 576]
        # deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims = [96, 192, 384, 768]
        self.stage_fuse_dims = [384, 768, 1536]  # 第二次聚合的维度

        # stage_Block  #用于将三个基分类器的第一个阶段进行融合
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # cur = 0
        # layer_scale_init_value = 1e-6
        # 考虑是否必要
        # self.frist_tumor_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        # self.frist_liver_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        # self.frist_original_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        self.frist_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.frist_dcn = nn.Sequential(
            # nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )

        self.second_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.second_dcn = nn.Sequential(
            # nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        # x_original_dims=[96, 192, 384, 768]
        # x_liver_dims=[96, 192, 384, 768]
        # x_tumor_dims=[96, 192, 384, 768]
        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(len(self.cross_attention_depths)):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i],
                                               dim_feedforward=4 * self.cross_attention_dim[i], nhead=8, dropout=0.1,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)

        self.cross_fuse_tumor_liver = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(2):
            stem = nn.Sequential(
                # nn.Linear(self.cross_fuse_dims[i], self.cross_attention_dim[i]),
                # nn.LayerNorm(self.cross_attention_dim[i])
                nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU(),
                # Block()
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList()  # 第一次三个拼起来后用stage_Block聚合
        for i in range(2):
            stem = nn.Sequential(
                nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),

                # LayerNorm(self.cross_attention_dim[i], eps=1e-6),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU()
            )
            self.cross_fuse_stagelayers.append(stem)

        self.downsample_layers = nn.ModuleList()
        for i in range(2):
            if i ==1:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(self.dims[i+1], 2*self.dims[i + 1], kernel_size=3, stride=2, padding=1)
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=3, stride=2, padding=1),
                )
            self.downsample_layers.append(downsample_layer)
        self.fuse_stagelayers = nn.ModuleList()
        for i in range(0,3):
            layer = nn.Sequential(
                nn.Conv2d(self.stage_fuse_dims[i], self.dims[i + 1], kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(self.dims[i+1]),
                # nn.GELU()
            )
            self.fuse_stagelayers.append(layer)

        self.fuse_stages = nn.ModuleList()
        for i in range(0,3):
            stage = nn.Sequential(*[Block(dim=self.dims[i + 1], ) for j in range(self.fuse_depths[i])]
                                  )
            self.fuse_stages.append(stage)

        self.classifier_weight = nn.Parameter(torch.ones(7), requires_grad=True)
        self.classifier = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6), nn.Linear(self.dims[3], num_classes))
        self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6), nn.Linear(self.dims[0], num_classes))
        self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6), nn.Linear(self.dims[1], num_classes))
        #self.head2 = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6), nn.Linear(self.dims[2], num_classes))
        self.head3 = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6), nn.Linear(self.dims[3], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        # self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        # self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):  # original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original, aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]
            x_liver, aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor, aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]
            # 检查每个层的输出是否有NAN
            # for i, output in enumerate([x_original,x_liver,x_tumor,aux_original,aux_liver,aux_tumor]):
            #     if torch.isnan(output).any():
            #         print(f'第{i}层输出出现了NAN!')
            #         break
        else:
            x_original = self.original_backbone(x)  # dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)  # dims=[64,128,256,512],
        # for i in range(len(self.cross_tumor_liver[0])):
        #     out1, out2 = self.cross_tumor_liver[0][i](out1, out2)
        feature = []
        # 这一步似乎没有必要,直接拼起来聚合特征就好，待考虑
        # x_tumor[0]=self.frist_tumor_stages(x_tumor[0])
        # x_liver[0] = self.frist_liver_stages(x_liver[0])
        # x_original[0] = self.frist_original_stages(x_original[0])
        x = self.frist_fuse(torch.cat((x_tumor[0], x_liver[0], x_original[0]), dim=1)).permute(0, 2, 3, 1)
        x = self.frist_dcn(x).permute(0, 3, 1, 2)
        feature.append(x)

        x = self.second_fuse(torch.cat((x_tumor[1], x_liver[1], x_original[1]), dim=1)).permute(0, 2, 3, 1)
        x = self.second_dcn(x).permute(0, 3, 1, 2)
        feature.append(x)

        for i in range(1,2):
            cross_out = []
            B, C, H, W = x_tumor[i + 2].shape
            x_tumor[i + 2] = x_tumor[i + 2].flatten(2).transpose(-1, -2)
            x_liver[i + 2] = x_liver[i + 2].flatten(2).transpose(-1, -2)
            x_original[i + 2] = x_original[i + 2].flatten(2).transpose(-1, -2)
            out1, out2 = x_tumor[i + 2], x_liver[i + 2]
            for layer in self.cross_tumor_liver[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_liver[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_tumor[i + 2], x_original[i + 2]
            for layer in self.cross_tumor_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_liver[i + 2], x_original[i + 2]
            for layer in self.cross_liver_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_liver_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            feature.append(self.cross_fuse_stagelayers[i](torch.cat(cross_out, dim=1)))
            # cross_out = []


        x = feature[0]
        x = self.downsample_layers[0](x)
        x = self.fuse_stagelayers[0](torch.cat((x, feature[1]), dim=1))
        x = self.fuse_stages[0](x)
        x = self.downsample_layers[1](x)
        x = self.fuse_stagelayers[2](torch.cat((x, feature[2]), dim=1))
        x = self.fuse_stages[2](x)

        # for i in range(3):
        #     down = self.downsample_layers[i](x)
        #     fuse = self.fuse_stagelayers[i](torch.cat((down, feature[i + 1]), dim=1))
        #     fuse = self.fuse_stages[i](fuse)
        #     x = fuse

        result = self.classifier(x.mean([-2, -1]))

        if self.aux_loss:
            head0 = self.head0(feature[0].mean([-2, -1]))
            head1 = self.head1(feature[1].mean([-2, -1]))
            #head2 = self.head2(feature[2].mean([-2, -1]))
            head3 = self.head3(feature[2].mean([-2, -1]))
            # weighted_pred = torch.zeros_like(result)
            weighted_pred = []
            for i, pre in enumerate([result, head0, head1, head3, aux_original, aux_liver, aux_tumor]):
                # weighted_pred += self.classifier_weight[i]*pre
                weighted_pred.append(self.classifier_weight[i] * pre)
            return weighted_pred

            # return [result,head0,head1,head2,head3,aux_original,aux_liver,aux_tumor]#,self.vote_weights
        else:
            return None#result

    def load_pretrained(self):
        tumor_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth', map_location="cpu")[
            'state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth', map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth', map_location="cpu")['model_state']

        self.tumor_backbone.load_state_dict(tumor_weight, strict=False)
        self.liver_backbone.load_state_dict(liver_weight, strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(
            in_features=self.original_backbone.head.projection.in_features,
            out_features=self.num_classes)
        del tumor_weight, liver_weight, original_weight
class liver_3_ablation_nostage3_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_ablation_nostage3_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2, 4, 32, 2], drop_path_rate=drop_path_rate,
                                    split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32], aux_loss=aux_loss, cross=False,
                                    **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,
                                       aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), drop_path_rate=drop_path_rate,
                                       aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        # 三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]

        self.cross_attention_depths = [3, 3]
        self.cross_attention_dim = [384, 768]  # 交叉注意力维度
        self.cross_fuse_dims = [768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        # self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288, 576]
        # deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims = [96, 192, 384, 768]
        self.stage_fuse_dims = [384, 768, 1536]  # 第二次聚合的维度

        # stage_Block  #用于将三个基分类器的第一个阶段进行融合
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # cur = 0
        # layer_scale_init_value = 1e-6
        # 考虑是否必要
        # self.frist_tumor_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        # self.frist_liver_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        # self.frist_original_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        self.frist_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.frist_dcn = nn.Sequential(
            # nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )

        self.second_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.second_dcn = nn.Sequential(
            # nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        # x_original_dims=[96, 192, 384, 768]
        # x_liver_dims=[96, 192, 384, 768]
        # x_tumor_dims=[96, 192, 384, 768]
        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(len(self.cross_attention_depths)):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i],
                                               dim_feedforward=4 * self.cross_attention_dim[i], nhead=8, dropout=0.1,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)

        self.cross_fuse_tumor_liver = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(2):
            stem = nn.Sequential(
                # nn.Linear(self.cross_fuse_dims[i], self.cross_attention_dim[i]),
                # nn.LayerNorm(self.cross_attention_dim[i])
                nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU(),
                # Block()
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList()  # 第一次三个拼起来后用stage_Block聚合
        for i in range(2):
            stem = nn.Sequential(
                nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),

                # LayerNorm(self.cross_attention_dim[i], eps=1e-6),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU()
            )
            self.cross_fuse_stagelayers.append(stem)

        self.downsample_layers = nn.ModuleList()
        for i in range(2):
            if i ==1:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=3, stride=2, padding=1),

                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=3, stride=2, padding=1),
                )
            self.downsample_layers.append(downsample_layer)
        self.fuse_stagelayers = nn.ModuleList()
        for i in range(0,2):
            layer = nn.Sequential(
                nn.Conv2d(self.stage_fuse_dims[i], self.dims[i + 1], kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(self.dims[i+1]),
                # nn.GELU()
            )
            self.fuse_stagelayers.append(layer)

        self.fuse_stages = nn.ModuleList()
        for i in range(0,2):
            stage = nn.Sequential(*[Block(dim=self.dims[i + 1], ) for j in range(self.fuse_depths[i])]
                                  )
            self.fuse_stages.append(stage)

        self.classifier_weight = nn.Parameter(torch.ones(7), requires_grad=True)
        self.classifier = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6), nn.Linear(self.dims[2], num_classes))
        self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6), nn.Linear(self.dims[0], num_classes))
        self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6), nn.Linear(self.dims[1], num_classes))
        self.head2 = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6), nn.Linear(self.dims[2], num_classes))
        #self.head3 = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6), nn.Linear(self.dims[3], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        # self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        # self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):  # original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original, aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]
            x_liver, aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor, aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]
            # 检查每个层的输出是否有NAN
            # for i, output in enumerate([x_original,x_liver,x_tumor,aux_original,aux_liver,aux_tumor]):
            #     if torch.isnan(output).any():
            #         print(f'第{i}层输出出现了NAN!')
            #         break
        else:
            x_original = self.original_backbone(x)  # dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)  # dims=[64,128,256,512],
        # for i in range(len(self.cross_tumor_liver[0])):
        #     out1, out2 = self.cross_tumor_liver[0][i](out1, out2)
        feature = []
        # 这一步似乎没有必要,直接拼起来聚合特征就好，待考虑
        # x_tumor[0]=self.frist_tumor_stages(x_tumor[0])
        # x_liver[0] = self.frist_liver_stages(x_liver[0])
        # x_original[0] = self.frist_original_stages(x_original[0])
        x = self.frist_fuse(torch.cat((x_tumor[0], x_liver[0], x_original[0]), dim=1)).permute(0, 2, 3, 1)
        x = self.frist_dcn(x).permute(0, 3, 1, 2)
        feature.append(x)

        x = self.second_fuse(torch.cat((x_tumor[1], x_liver[1], x_original[1]), dim=1)).permute(0, 2, 3, 1)
        x = self.second_dcn(x).permute(0, 3, 1, 2)
        feature.append(x)

        for i in range(0,1):
            cross_out = []
            B, C, H, W = x_tumor[i + 2].shape
            x_tumor[i + 2] = x_tumor[i + 2].flatten(2).transpose(-1, -2)
            x_liver[i + 2] = x_liver[i + 2].flatten(2).transpose(-1, -2)
            x_original[i + 2] = x_original[i + 2].flatten(2).transpose(-1, -2)
            out1, out2 = x_tumor[i + 2], x_liver[i + 2]
            for layer in self.cross_tumor_liver[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_liver[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_tumor[i + 2], x_original[i + 2]
            for layer in self.cross_tumor_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_liver[i + 2], x_original[i + 2]
            for layer in self.cross_liver_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_liver_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            feature.append(self.cross_fuse_stagelayers[i](torch.cat(cross_out, dim=1)))
            # cross_out = []


        x = feature[0]
        x = self.downsample_layers[0](x)
        x = self.fuse_stagelayers[0](torch.cat((x, feature[1]), dim=1))
        x = self.fuse_stages[0](x)
        x = self.downsample_layers[1](x)
        x = self.fuse_stagelayers[1](torch.cat((x, feature[2]), dim=1))
        x = self.fuse_stages[1](x)

        # for i in range(3):
        #     down = self.downsample_layers[i](x)
        #     fuse = self.fuse_stagelayers[i](torch.cat((down, feature[i + 1]), dim=1))
        #     fuse = self.fuse_stages[i](fuse)
        #     x = fuse

        result = self.classifier(x.mean([-2, -1]))

        if self.aux_loss:
            head0 = self.head0(feature[0].mean([-2, -1]))
            head1 = self.head1(feature[1].mean([-2, -1]))
            head2 = self.head2(feature[2].mean([-2, -1]))
            #head3 = self.head3(feature[2].mean([-2, -1]))
            # weighted_pred = torch.zeros_like(result)
            weighted_pred = []
            for i, pre in enumerate([result, head0, head1, head2, aux_original, aux_liver, aux_tumor]):
                # weighted_pred += self.classifier_weight[i]*pre
                weighted_pred.append(self.classifier_weight[i] * pre)
            return weighted_pred

            # return [result,head0,head1,head2,head3,aux_original,aux_liver,aux_tumor]#,self.vote_weights
        else:
            return None#result

    def load_pretrained(self):
        tumor_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth', map_location="cpu")[
            'state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth', map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth', map_location="cpu")['model_state']

        self.tumor_backbone.load_state_dict(tumor_weight, strict=False)
        self.liver_backbone.load_state_dict(liver_weight, strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(
            in_features=self.original_backbone.head.projection.in_features,
            out_features=self.num_classes)
        del tumor_weight, liver_weight, original_weight

class liver_3_ablation_nomulitscale_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_ablation_nomulitscale_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=drop_path_rate,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=False, **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3),drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        #三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]

        self.cross_attention_depths=[3,3]
        self.cross_attention_dim = [384,768] #交叉注意力维度
        self.cross_fuse_dims = [768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        #self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288,576]
        #deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims=[96, 192, 384,768]
        self.stage_fuse_dims = [ 384, 768,1536]#第二次聚合的维度

        # stage_Block  #用于将三个基分类器的第一个阶段进行融合
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # cur = 0
        # layer_scale_init_value = 1e-6
        # 考虑是否必要
        # self.frist_tumor_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        # self.frist_liver_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        # self.frist_original_stages= nn.Sequential(
        #         *[Block(dim=self.frist_dims[0], drop_path=dp_rates[cur + j],
        #         layer_scale_init_value=layer_scale_init_value) for j in range(self.frist_depths[0])]
        #     )
        self.frist_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.frist_dcn = nn.Sequential(
            #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[0] // 3,groups=self.fuse_dcn_dims[0] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )

        self.second_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.second_dcn = nn.Sequential(
            #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[1] // 3,groups=self.fuse_dcn_dims[1] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        # x_original_dims=[96, 192, 384, 768]
        # x_liver_dims=[96, 192, 384, 768]
        # x_tumor_dims=[96, 192, 384, 768]
        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(len(self.cross_attention_depths)):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i],dim_feedforward=4*self.cross_attention_dim[i], nhead=8, dropout=0.1,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)


        self.cross_fuse_tumor_liver = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(2):
            stem = nn.Sequential(
                # nn.Linear(self.cross_fuse_dims[i], self.cross_attention_dim[i]),
                # nn.LayerNorm(self.cross_attention_dim[i])
                nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1,padding=0),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU(),
                # Block()
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(2):
            stem = nn.Sequential(
                nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),

                #LayerNorm(self.cross_attention_dim[i], eps=1e-6),
                # nn.BatchNorm2d(self.cross_attention_dim[i]),
                # nn.GELU()
            )
            self.cross_fuse_stagelayers.append(stem)


        # self.downsample_layers= nn.ModuleList()
        # for i in range(3):
        #     downsample_layer = nn.Sequential(
        #         nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, stride=2,padding=1),
        #     )
        #     self.downsample_layers.append(downsample_layer)
        # self.fuse_stagelayers= nn.ModuleList()
        # for i in range(3):
        #     layer = nn.Sequential(
        #         nn.Conv2d(self.stage_fuse_dims[i], self.dims[i+1], kernel_size=3, stride=1,padding=1),
        #         # nn.BatchNorm2d(self.dims[i+1]),
        #         # nn.GELU()
        #     )
        #     self.fuse_stagelayers.append(layer)
        #
        # self.fuse_stages = nn.ModuleList()
        # for i in range(3):
        #     stage = nn.Sequential(*[Block(dim=self.dims[i+1],drop_path=0.1) for j in range(self.fuse_depths[i])]
        #     )
        #     self.fuse_stages.append(stage)

        self.classifier_weight = nn.Parameter(torch.full(size=(7,), fill_value=0.125),requires_grad=True)
        # self.classifier = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6),nn.Linear(self.dims[0], num_classes))
        self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6),nn.Linear(self.dims[1], num_classes))
        self.head2 = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6),nn.Linear(self.dims[2], num_classes))
        self.head3 = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        #self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):#original_img,liver_img,tumor_img
        # visual_feature=[]
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]
            x_liver,aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor,aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]
            # visual_feature.append((x_original,x_liver,x_tumor))
            # B, C, H, W = x_original[2].shape
            # for i in range(B):
            #     image = x_original[2][i].detach().cpu().numpy().transpose(1, 2, 0)
            #     if True:
            #         image = image.sum(-1)
            #         fig = plt.figure()
            #         ax = fig.add_subplot(1, 1, 1)
            #         ax.imshow(image)
            #         plt.axis('off')
            #         plt.savefig(f'/home/uax/SCY/LiverClassification/feature_map/{i}_{0}.png')
            #         plt.close()
            #     else:
            #         for c in range(C):
            #             feature_map_c=image[:,:,c]
            #             fig = plt.figure()
            #             ax = fig.add_subplot(1,1,1)
            #             ax.imshow(feature_map_c)
            #             plt.axis('off')
            #             plt.savefig(f'/home/uax/SCY/LiverClassification/feature_map/{i}_{c}.png')
            #             plt.close()
            # a=0
        else:
            x_original = self.original_backbone(x)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)#dims=[64,128,256,512],
        # for i in range(len(self.cross_tumor_liver[0])):
        #     out1, out2 = self.cross_tumor_liver[0][i](out1, out2)
        feature = []
        #这一步似乎没有必要,直接拼起来聚合特征就好，待考虑
        # x_tumor[0]=self.frist_tumor_stages(x_tumor[0])
        # x_liver[0] = self.frist_liver_stages(x_liver[0])
        # x_original[0] = self.frist_original_stages(x_original[0])
        x=self.frist_fuse(torch.cat((x_tumor[0],x_liver[0],x_original[0]),dim=1)).permute(0,2,3,1)
        x=self.frist_dcn(x).permute(0,3,1,2)
        feature.append(x)

        x=self.second_fuse(torch.cat((x_tumor[1],x_liver[1],x_original[1]),dim=1)).permute(0,2,3,1)
        x=self.second_dcn(x).permute(0,3,1,2)
        feature.append(x)
        #.transpose(-1, -2)

        for i in range(2):
            cross_out = []
            B,C,H,W = x_tumor[i+2].shape
            x_tumor[i+2]= x_tumor[i+2].flatten(2).transpose(-1, -2)
            x_liver[i + 2]=x_liver[i+2].flatten(2).transpose(-1, -2)
            x_original[i + 2]=x_original[i + 2].flatten(2).transpose(-1, -2)
            out1, out2 = x_tumor[i + 2], x_liver[i + 2]
            for layer in self.cross_tumor_liver[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_liver[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_tumor[i + 2], x_original[i + 2]
            for layer in self.cross_tumor_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            out1, out2 = x_liver[i + 2], x_original[i + 2]
            for layer in self.cross_liver_original[i]:
                out1, out2 = layer(out1, out2)
                # if torch.isnan(out1).any() or torch.isnan(out2).any():
                #     print(f'第{i}层输出出现了NAN!')
            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_liver_original[i](torch.cat((out1, out2), dim=1)))
            # cross_out.append(out1)
            # cross_out.append(out2)
            feature.append(self.cross_fuse_stagelayers[i](torch.cat(cross_out, dim=1)))
            #cross_out = []

        # x = feature[0]
        # for i in range(3):
        #     down = self.downsample_layers[i](x)
        #     fuse = self.fuse_stagelayers[i](torch.cat((down, feature[i + 1]), dim=1))
        #     fuse = self.fuse_stages[i](fuse)
        #     x = fuse
        #
        #
        # mulit_scale_head = self.classifier(x.mean([-2, -1]))


        if self.aux_loss:
            head0 = self.head0(feature[0].mean([-2, -1]))
            head1 = self.head1(feature[1].mean([-2, -1]))
            head2 = self.head2(feature[2].mean([-2, -1]))
            head3 = self.head3(feature[3].mean([-2, -1]))
            #weighted_pred = torch.zeros_like(result)
            weighted_pred = []
            for i ,pre in enumerate([head0,head1,head2,head3,aux_original,aux_liver,aux_tumor]):
                #weighted_pred += self.classifier_weight[i]*pre
                weighted_pred.append(self.classifier_weight[i]*pre)
            return weighted_pred

            #return [result,head0,head1,head2,head3,aux_original,aux_liver,aux_tumor]#,self.vote_weights
        else:
            return None

    def load_pretrained(self):
        tumor_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth',map_location="cpu")['state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth',map_location="cpu")['model_state']
        # self.tumor_backbone.head.out_features = self.num_classes
        # self.liver_backbone.head.out_features = self.num_classes
        # self.original_backbone.head.projection.out_features = self.num_classes
        self.tumor_backbone.load_state_dict(tumor_weight,strict=False)
        self.liver_backbone.load_state_dict(liver_weight,strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features, out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(in_features=self.original_backbone.head.projection.in_features,
                                             out_features=self.num_classes)
        del tumor_weight,liver_weight,original_weight
class liver_3_ablation_nostage01_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_ablation_nostage01_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=drop_path_rate,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=False, **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3),drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        #三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]

        self.cross_attention_depths=[3,3]
        self.cross_attention_dim = [384,768] #交叉注意力维度
        self.cross_fuse_dims = [768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        #self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288,576]
        #deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims=[96, 192, 384,768]
        self.stage_fuse_dims = [ 384, 768,1536]#第二次聚合的维度

        # self.frist_fuse = nn.Sequential(
        #     nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
        #
        # )
        # self.frist_dcn = nn.Sequential(
        #     InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[0] // 3,groups=self.fuse_dcn_dims[0] // 3),
        #     InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
        # )

        # self.second_fuse = nn.Sequential(
        #     nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
        # )
        # self.second_dcn = nn.Sequential(
        #     InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[1] // 3,groups=self.fuse_dcn_dims[1] // 3),
        #     InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
        # )

        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(len(self.cross_attention_depths)):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i],dim_feedforward=4*self.cross_attention_dim[i], nhead=8, dropout=0.1,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)


        self.cross_fuse_tumor_liver = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(2):
            stem = nn.Sequential(
                nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1,padding=0),
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(2):
            stem = nn.Sequential(
                nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),
            )
            self.cross_fuse_stagelayers.append(stem)


        self.downsample_layers= nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, stride=2,padding=1),
            )
            self.downsample_layers.append(downsample_layer)
        self.fuse_stagelayers= nn.ModuleList()
        for i in range(3):
            layer = nn.Sequential(
                nn.Conv2d(self.stage_fuse_dims[i], self.dims[i+1], kernel_size=3, stride=1,padding=1),

            )
            self.fuse_stagelayers.append(layer)

        self.fuse_stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(*[Block(dim=self.dims[i+1],) for j in range(self.fuse_depths[i])]
            )
            self.fuse_stages.append(stage)

        self.classifier_weight = nn.Parameter(torch.ones(8),requires_grad=True)
        self.classifier = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6),nn.Linear(self.dims[0], num_classes))
        self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6),nn.Linear(self.dims[1], num_classes))
        self.head2 = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6),nn.Linear(self.dims[2], num_classes))
        self.head3 = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        #self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]

            x_liver,aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor,aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]

        else:
            x_original = self.original_backbone(x)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)#dims=[64,128,256,512],

        feature = []
        # x=self.frist_fuse(torch.cat((x_tumor[0],x_liver[0],x_original[0]),dim=1)).permute(0,2,3,1)
        # x=self.frist_dcn(x).permute(0,3,1,2)
        # feature.append(x)
        feature.append('x')

        # x=self.second_fuse(torch.cat((x_tumor[1],x_liver[1],x_original[1]),dim=1)).permute(0,2,3,1)
        # x=self.second_dcn(x).permute(0,3,1,2)
        feature.append('x')


        for i in range(2):
            cross_out = []
            B,C,H,W = x_tumor[i+2].shape
            x_tumor[i+2]= x_tumor[i+2].flatten(2).transpose(-1, -2)
            x_liver[i + 2]=x_liver[i+2].flatten(2).transpose(-1, -2)
            x_original[i + 2]=x_original[i + 2].flatten(2).transpose(-1, -2)
            out1, out2 = x_tumor[i + 2], x_liver[i + 2]
            for layer in self.cross_tumor_liver[i]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_liver[i](torch.cat((out1, out2), dim=1)))

            out1, out2 = x_tumor[i + 2], x_original[i + 2]
            for layer in self.cross_tumor_original[i]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_original[i](torch.cat((out1, out2), dim=1)))

            out1, out2 = x_liver[i + 2], x_original[i + 2]
            for layer in self.cross_liver_original[i]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_liver_original[i](torch.cat((out1, out2), dim=1)))

            feature.append(self.cross_fuse_stagelayers[i](torch.cat(cross_out, dim=1)))

        x = feature[2]
        for i in range(2,3):
            down = self.downsample_layers[i](x)
            fuse = self.fuse_stagelayers[i](torch.cat((down, feature[i + 1]), dim=1))
            fuse = self.fuse_stages[i](fuse)
            x = fuse


        result = self.classifier(x.mean([-2, -1]))


        if self.aux_loss:
            #head0 = self.head0(feature[0].mean([-2, -1]))
            # head1 = self.head1(feature[1].mean([-2, -1]))
            head2 = self.head2(feature[2].mean([-2, -1]))
            head3 = self.head3(feature[3].mean([-2, -1]))

            weighted_pred = []
            for i ,pre in enumerate([result,head2,head3,aux_original,aux_liver,aux_tumor]):

                weighted_pred.append(self.classifier_weight[i]*pre)
            return weighted_pred

        else:
            return result

    def load_pretrained(self):
        tumor_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth',map_location="cpu")['state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth',map_location="cpu")['model_state']

        self.tumor_backbone.load_state_dict(tumor_weight,strict=False)
        self.liver_backbone.load_state_dict(liver_weight,strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features, out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(in_features=self.original_backbone.head.projection.in_features,
                                             out_features=self.num_classes)
        del tumor_weight,liver_weight,original_weight


class liver_3_ablation_nostage012_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_ablation_nostage012_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=drop_path_rate,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=False, **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3),drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        #三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]

        self.cross_attention_depths=[3,3]
        self.cross_attention_dim = [384,768] #交叉注意力维度
        self.cross_fuse_dims = [768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        #self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288,576]
        #deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims=[96, 192, 384,768]
        self.stage_fuse_dims = [ 384, 768,1536]#第二次聚合的维度

        # self.frist_fuse = nn.Sequential(
        #     nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
        #
        # )
        # self.frist_dcn = nn.Sequential(
        #     InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[0] // 3,groups=self.fuse_dcn_dims[0] // 3),
        #     InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
        # )

        # self.second_fuse = nn.Sequential(
        #     nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
        # )
        # self.second_dcn = nn.Sequential(
        #     InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[1] // 3,groups=self.fuse_dcn_dims[1] // 3),
        #     InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
        # )

        self.cross_tumor_liver = nn.ModuleList()
        self.cross_tumor_original = nn.ModuleList()
        self.cross_liver_original = nn.ModuleList()
        for i in range(1,2):
            stage_modules = nn.ModuleList()
            for j in range(self.cross_attention_depths[i]):
                layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i],dim_feedforward=4*self.cross_attention_dim[i], nhead=8, dropout=0.1,
                                               normalize_before=False)
                stage_modules.append(layer)
            self.cross_tumor_liver.append(stage_modules)
            self.cross_tumor_original.append(stage_modules)
            self.cross_liver_original.append(stage_modules)


        self.cross_fuse_tumor_liver = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.cross_fuse_tumor_original = nn.ModuleList()
        self.cross_fuse_liver_original = nn.ModuleList()
        for i in range(1,2):
            stem = nn.Sequential(
                nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1,padding=0),
            )
            self.cross_fuse_tumor_liver.append(stem)
            self.cross_fuse_tumor_original.append(stem)
            self.cross_fuse_liver_original.append(stem)

        self.cross_fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        for i in range(1,2):
            stem = nn.Sequential(
                nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),
            )
            self.cross_fuse_stagelayers.append(stem)


        # self.downsample_layers= nn.ModuleList()
        # for i in range(3):
        #     downsample_layer = nn.Sequential(
        #         nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, stride=2,padding=1),
        #     )
        #     self.downsample_layers.append(downsample_layer)
        # self.fuse_stagelayers= nn.ModuleList()
        # for i in range(3):
        #     layer = nn.Sequential(
        #         nn.Conv2d(self.stage_fuse_dims[i], self.dims[i+1], kernel_size=3, stride=1,padding=1),
        #
        #     )
        #     self.fuse_stagelayers.append(layer)
        #
        # self.fuse_stages = nn.ModuleList()
        # for i in range(3):
        #     stage = nn.Sequential(*[Block(dim=self.dims[i+1],) for j in range(self.fuse_depths[i])]
        #     )
        #     self.fuse_stages.append(stage)

        self.classifier_weight = nn.Parameter(torch.ones(4),requires_grad=True)
        self.classifier = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        # self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6),nn.Linear(self.dims[0], num_classes))
        # self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6),nn.Linear(self.dims[1], num_classes))
        # self.head2 = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6),nn.Linear(self.dims[2], num_classes))
        # self.head3 = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        #self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]

            x_liver,aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor,aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]

        else:
            x_original = self.original_backbone(x)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)#dims=[64,128,256,512],

        feature = []
        # x=self.frist_fuse(torch.cat((x_tumor[0],x_liver[0],x_original[0]),dim=1)).permute(0,2,3,1)
        # x=self.frist_dcn(x).permute(0,3,1,2)
        # feature.append(x)
        feature.append('x')

        # x=self.second_fuse(torch.cat((x_tumor[1],x_liver[1],x_original[1]),dim=1)).permute(0,2,3,1)
        # x=self.second_dcn(x).permute(0,3,1,2)
        feature.append('x')


        for i in range(1,2):
            cross_out = []
            B,C,H,W = x_tumor[i+2].shape
            x_tumor[i+2]= x_tumor[i+2].flatten(2).transpose(-1, -2)
            x_liver[i + 2]=x_liver[i+2].flatten(2).transpose(-1, -2)
            x_original[i + 2]=x_original[i + 2].flatten(2).transpose(-1, -2)
            out1, out2 = x_tumor[i + 2], x_liver[i + 2]
            for layer in self.cross_tumor_liver[i-1]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_liver[i-1](torch.cat((out1, out2), dim=1)))

            out1, out2 = x_tumor[i + 2], x_original[i + 2]
            for layer in self.cross_tumor_original[i-1]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_tumor_original[i-1](torch.cat((out1, out2), dim=1)))

            out1, out2 = x_liver[i + 2], x_original[i + 2]
            for layer in self.cross_liver_original[i-1]:
                out1, out2 = layer(out1, out2)

            out1 = out1.transpose(1, 2).view(B, C, H, W).contiguous()
            out2 = out2.transpose(1, 2).view(B, C, H, W).contiguous()
            cross_out.append(self.cross_fuse_liver_original[i-1](torch.cat((out1, out2), dim=1)))

            feature.append(self.cross_fuse_stagelayers[i-1](torch.cat(cross_out, dim=1)))

        x = feature[-1]
        # for i in range(2,3):
        #     down = self.downsample_layers[i](x)
        #     fuse = self.fuse_stagelayers[i](torch.cat((down, feature[i + 1]), dim=1))
        #     fuse = self.fuse_stages[i](fuse)
        #     x = fuse


        result = self.classifier(x.mean([-2, -1]))


        if self.aux_loss:
            #head0 = self.head0(feature[0].mean([-2, -1]))
            # head1 = self.head1(feature[1].mean([-2, -1]))
            # head2 = self.head2(feature[2].mean([-2, -1]))
            # head3 = self.head3(feature[3].mean([-2, -1]))

            weighted_pred = []
            for i ,pre in enumerate([result,aux_original,aux_liver,aux_tumor]):

                weighted_pred.append(self.classifier_weight[i]*pre)
            return weighted_pred

        else:
            return result

    def load_pretrained(self):
        tumor_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth',map_location="cpu")['state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth',map_location="cpu")['model_state']

        self.tumor_backbone.load_state_dict(tumor_weight,strict=False)
        self.liver_backbone.load_state_dict(liver_weight,strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features, out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(in_features=self.original_backbone.head.projection.in_features,
                                             out_features=self.num_classes)
        del tumor_weight,liver_weight,original_weight


class liver_3_ablation_nostage0123_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_ablation_nostage0123_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=drop_path_rate,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=False, **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3),drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        #三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]

        self.cross_attention_depths=[3,3]
        self.cross_attention_dim = [384,768] #交叉注意力维度
        self.cross_fuse_dims = [768, 1536]  # 交叉注意力两两concat
        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        #self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288,576]
        #deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims=[96, 192, 384,768]
        self.stage_fuse_dims = [ 384, 768,1536]#第二次聚合的维度

        # self.frist_fuse = nn.Sequential(
        #     nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
        #
        # )
        # self.frist_dcn = nn.Sequential(
        #     InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[0] // 3,groups=self.fuse_dcn_dims[0] // 3),
        #     InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
        # )

        # self.second_fuse = nn.Sequential(
        #     nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
        # )
        # self.second_dcn = nn.Sequential(
        #     InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[1] // 3,groups=self.fuse_dcn_dims[1] // 3),
        #     InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
        # )
        #
        # self.cross_tumor_liver = nn.ModuleList()
        # self.cross_tumor_original = nn.ModuleList()
        # self.cross_liver_original = nn.ModuleList()
        # for i in range(1,2):
        #     stage_modules = nn.ModuleList()
        #     for j in range(self.cross_attention_depths[i]):
        #         layer = Cross_3_AttentionBlock(d_model=self.cross_attention_dim[i],dim_feedforward=4*self.cross_attention_dim[i], nhead=8, dropout=0.1,
        #                                        normalize_before=False)
        #         stage_modules.append(layer)
        #     self.cross_tumor_liver.append(stage_modules)
        #     self.cross_tumor_original.append(stage_modules)
        #     self.cross_liver_original.append(stage_modules)
        #
        #
        # self.cross_fuse_tumor_liver = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # self.cross_fuse_tumor_original = nn.ModuleList()
        # self.cross_fuse_liver_original = nn.ModuleList()
        # for i in range(1,2):
        #     stem = nn.Sequential(
        #         nn.Conv2d(self.cross_fuse_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1,padding=0),
        #     )
        #     self.cross_fuse_tumor_liver.append(stem)
        #     self.cross_fuse_tumor_original.append(stem)
        #     self.cross_fuse_liver_original.append(stem)
        #
        # self.cross_fuse_stagelayers = nn.ModuleList() #第一次三个拼起来后用stage_Block聚合
        # for i in range(1,2):
        #     stem = nn.Sequential(
        #         nn.Conv2d(self.fuse_stage_dims[i], self.cross_attention_dim[i], kernel_size=1, stride=1, padding=0),
        #     )
        #     self.cross_fuse_stagelayers.append(stem)


        # self.downsample_layers= nn.ModuleList()
        # for i in range(3):
        #     downsample_layer = nn.Sequential(
        #         nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=3, stride=2,padding=1),
        #     )
        #     self.downsample_layers.append(downsample_layer)
        # self.fuse_stagelayers= nn.ModuleList()
        # for i in range(3):
        #     layer = nn.Sequential(
        #         nn.Conv2d(self.stage_fuse_dims[i], self.dims[i+1], kernel_size=3, stride=1,padding=1),
        #
        #     )
        #     self.fuse_stagelayers.append(layer)
        #
        # self.fuse_stages = nn.ModuleList()
        # for i in range(3):
        #     stage = nn.Sequential(*[Block(dim=self.dims[i+1],) for j in range(self.fuse_depths[i])]
        #     )
        #     self.fuse_stages.append(stage)

        self.classifier_weight = nn.Parameter(torch.ones(3),requires_grad=True)
        # self.classifier = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        # self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6),nn.Linear(self.dims[0], num_classes))
        # self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6),nn.Linear(self.dims[1], num_classes))
        # self.head2 = nn.Sequential(nn.LayerNorm(self.dims[2], eps=1e-6),nn.Linear(self.dims[2], num_classes))
        # self.head3 = nn.Sequential(nn.LayerNorm(self.dims[3], eps=1e-6),nn.Linear(self.dims[3], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        #self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]

            x_liver,aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor,aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]

        else:
            x_original = self.original_backbone(x)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)#dims=[64,128,256,512],



        if self.aux_loss:
            #head0 = self.head0(feature[0].mean([-2, -1]))
            # head1 = self.head1(feature[1].mean([-2, -1]))
            # head2 = self.head2(feature[2].mean([-2, -1]))
            # head3 = self.head3(feature[3].mean([-2, -1]))

            weighted_pred = []
            for i ,pre in enumerate([aux_original,aux_liver,aux_tumor]):

                weighted_pred.append(self.classifier_weight[i]*pre)
            return weighted_pred

        else:
            return None

    def load_pretrained(self):
        tumor_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth',map_location="cpu")['state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth',map_location="cpu")['model_state']

        self.tumor_backbone.load_state_dict(tumor_weight,strict=False)
        self.liver_backbone.load_state_dict(liver_weight,strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features, out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(in_features=self.original_backbone.head.projection.in_features,
                                             out_features=self.num_classes)
        del tumor_weight,liver_weight,original_weight

class liver_3_ablation_noglobal_net(nn.Module):
    def __init__(self,num_classes=2,aux_loss=False,drop_path_rate=0.2,**kwargs):
        super(liver_3_ablation_noglobal_net,self).__init__()
        self.aux_loss=aux_loss
        self.num_classes = num_classes

        self.tumor_backbone = CSWin(embed_dim=96, depth=[2,4,32,2],drop_path_rate=drop_path_rate,
        split_size=[1,2,7,7], num_heads=[4,8,16,32],aux_loss=aux_loss,cross=False, **kwargs)
        self.liver_backbone = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        self.original_backbone = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3),drop_path_rate=drop_path_rate,aux_loss=aux_loss, **kwargs)
        # self.pixel_decoder = MSDeformAttnPixelDecoder(feature_channels=[96, 192, 384, 768],transformer_in_channels=[192, 384, 768],
        #                      transformer_dim_feedforward=1024,transformer_enc_layers=6,)

        #三个backbone的维度
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]
        # dims=[96, 192, 384, 768]


        self.fuse_stage_dims = [1152, 2304]  # 交叉注意力每一个阶段concat
        #self.fuse_stage_6_dims = [1152, 2304, 4608]
        self.fuse_dcn_dims = [288,576]
        #deformed之后的特征金子塔
        self.fuse_depths = [2, 2, 2]
        self.dims=[96, 192, 384,768]
        self.stage_fuse_dims = [ 384, 768,1536]#第二次聚合的维度


        self.frist_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[0], self.fuse_dcn_dims[0] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.frist_dcn = nn.Sequential(
            #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[0] // 3,groups=self.fuse_dcn_dims[0] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[0] // 3, groups=self.fuse_dcn_dims[0] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )

        self.second_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_dcn_dims[1], self.fuse_dcn_dims[1] // 3, kernel_size=1, stride=1, padding=0),
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )
        self.second_dcn = nn.Sequential(
            #nn.Conv2d(self.frist_dims, self.frist_dims // 3, kernel_size=1, stride=1, padding=0),
            InternImageLayer(DCNv3,channels=self.fuse_dcn_dims[1] // 3,groups=self.fuse_dcn_dims[1] // 3),
            InternImageLayer(DCNv3, channels=self.fuse_dcn_dims[1] // 3, groups=self.fuse_dcn_dims[1] // 3)
            # Block(dim=self.frist_dims),Block(dim=self.frist_dims),
            # nn.BatchNorm2d(self.frist_dims//3),
            # nn.GELU()
        )


        self.classifier_weight = nn.Parameter(torch.full(size=(5,), fill_value=0.2),requires_grad=True)
        self.head0 = nn.Sequential(nn.LayerNorm(self.dims[0], eps=1e-6),nn.Linear(self.dims[0], num_classes))
        self.head1 = nn.Sequential(nn.LayerNorm(self.dims[1], eps=1e-6),nn.Linear(self.dims[1], num_classes))
        self.apply(self._init_weights)
        self.load_pretrained()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

        #self.loss_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))
        #self.vote_weights = nn.Parameter(torch.Tensor([1, 1, 1, 1]))

    def forward(self, x):#original_img,liver_img,tumor_img
        if self.aux_loss:
            x_original,aux_original = self.original_backbone(x)  # dims=[112, 224, 448, 896]
            x_liver,aux_liver = self.liver_backbone(x)  # dims=[128, 256, 512, 1024]
            x_tumor,aux_tumor = self.tumor_backbone(x)  # dims=[96, 192, 384, 768]
            # 检查每个层的输出是否有NAN
            # for i, output in enumerate([x_original,x_liver,x_tumor,aux_original,aux_liver,aux_tumor]):
            #     if torch.isnan(output).any():
            #         print(f'第{i}层输出出现了NAN!')
            #         break
        else:
            x_original = self.original_backbone(x)#dims=[96, 192, 384, 768]
            x_liver = self.liver_backbone(x)#dims=[128, 256, 512, 1024]
            x_tumor = self.tumor_backbone(x)#dims=[64,128,256,512],
        # for i in range(len(self.cross_tumor_liver[0])):
        #     out1, out2 = self.cross_tumor_liver[0][i](out1, out2)
        feature = []
        #这一步似乎没有必要,直接拼起来聚合特征就好，待考虑
        # x_tumor[0]=self.frist_tumor_stages(x_tumor[0])
        # x_liver[0] = self.frist_liver_stages(x_liver[0])
        # x_original[0] = self.frist_original_stages(x_original[0])
        x=self.frist_fuse(torch.cat((x_tumor[0],x_liver[0],x_original[0]),dim=1)).permute(0,2,3,1)
        x=self.frist_dcn(x).permute(0,3,1,2)
        feature.append(x)

        x=self.second_fuse(torch.cat((x_tumor[1],x_liver[1],x_original[1]),dim=1)).permute(0,2,3,1)
        x=self.second_dcn(x).permute(0,3,1,2)
        feature.append(x)
        #.transpose(-1, -2)



        if self.aux_loss:
            head0 = self.head0(feature[0].mean([-2, -1]))
            head1 = self.head1(feature[1].mean([-2, -1]))

            #weighted_pred = torch.zeros_like(result)
            weighted_pred = []
            for i ,pre in enumerate([head0,head1,aux_original,aux_liver,aux_tumor]):
                #weighted_pred += self.classifier_weight[i]*pre
                weighted_pred.append(self.classifier_weight[i]*pre)
            return weighted_pred

            #return [result,head0,head1,head2,head3,aux_original,aux_liver,aux_tumor]#,self.vote_weights
        else:
            return None

    def load_pretrained(self):
        tumor_weight=torch.load(r'/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth',map_location="cpu")['state_dict_ema']
        liver_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth',map_location="cpu")['model']
        original_weight = torch.load(r'/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth',map_location="cpu")['model_state']
        # self.tumor_backbone.head.out_features = self.num_classes
        # self.liver_backbone.head.out_features = self.num_classes
        # self.original_backbone.head.projection.out_features = self.num_classes
        self.tumor_backbone.load_state_dict(tumor_weight,strict=False)
        self.liver_backbone.load_state_dict(liver_weight,strict=False)
        self.original_backbone.load_state_dict(original_weight, strict=False)
        self.tumor_backbone.head = nn.Linear(in_features=self.tumor_backbone.head.in_features, out_features=self.num_classes)
        self.liver_backbone.head = nn.Linear(in_features=self.liver_backbone.head.in_features,
                                             out_features=self.num_classes)
        self.original_backbone.head.projection = nn.Linear(in_features=self.original_backbone.head.projection.in_features,
                                             out_features=self.num_classes)
        del tumor_weight,liver_weight,original_weight