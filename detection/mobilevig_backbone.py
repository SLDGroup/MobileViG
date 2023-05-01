import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

import numpy

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


'''
@article{han2022vision,
  title={Vision GNN: An Image is Worth Graph of Nodes},
  author={Han, Kai and Wang, Yunhe and Guo, Jianyuan and Tang, Yehui and Wu, Enhua},
  journal={arXiv preprint arXiv:2206.00272},
  year={2022}
}
'''


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'mobilevig': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}
    
class Stem(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.GELU):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU()   
        )
        
    def forward(self, x):
        return self.stem(x)
    
class MLP(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x

class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    
    K is the number of superpatches, therefore hops equals res // K.
    """
    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
            )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape
            
        x_j = x - x
        for i in range(self.K, H, self.K):
            x_c = x - torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)
            x_j = torch.max(x_j, x_c)
        for i in range(self.K, W, self.K):
            x_r = x - torch.cat([x[:, :, :, -i:], x[:, :, :, :-i]], dim=3)
            x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, drop_path=0.0, K=2):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels, in_channels * 2, K=self.K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  # out_channels back to 1x}
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

       
    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp

        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features # same as input
        hidden_features = hidden_features or in_features # x4
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class MobileViG(torch.nn.Module):
    def __init__(self, local_blocks, local_channels,
                 global_blocks, global_channels,
                 dropout=0., drop_path=0., emb_dims=512,
                 K=2, distillation=True, num_classes=1000,
                 pretrained=None, out_indices=None):
        super(MobileViG, self).__init__()

        self.distillation = distillation
        self.out_indices = out_indices
        self.pretrained = pretrained
        
        n_blocks = sum(global_blocks) + sum(local_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule 
        dpr_idx = 0

        self.stem = Stem(input_dim=3, output_dim=local_channels[0])
        
        # local processing with inverted residuals
        self.local_backbone = nn.ModuleList([])
        for i in range(len(local_blocks)):
            if i > 0:
                self.local_backbone.append(Downsample(local_channels[i-1], local_channels[i]))
            for _ in range(local_blocks[i]):
                self.local_backbone.append(InvertedResidual(dim=local_channels[i], mlp_ratio=4, drop_path=dpr[dpr_idx]))
                dpr_idx += 1
        self.local_backbone.append(Downsample(local_channels[-1], global_channels[0]))  # transition from local to global

        # global processing with svga
        self.backbone = nn.ModuleList([])
        for i in range(len(global_blocks)):
            if i > 0:
                self.backbone.append(Downsample(global_channels[i-1], global_channels[i]))
            for j in range(global_blocks[i]):
                self.backbone += [nn.Sequential(
                                    Grapher(global_channels[i], drop_path=dpr[dpr_idx], K=K),
                                    FFN(global_channels[i], global_channels[i] * 4, drop_path=dpr[dpr_idx]))
                                    ]
                dpr_idx += 1

        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)


    def init_weights(self):
        logger = get_root_logger()
        print("Pretrained weights being loaded")
        logger.warn('Pretrained weights being loaded')
        ckpt_path = self.pretrained
        ckpt = _load_checkpoint(
            ckpt_path, logger=logger, map_location='cpu')
        print("ckpt keys: ", ckpt.keys())
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict_ema']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt

        state_dict = _state_dict
        missing_keys, unexpected_keys = \
            self.load_state_dict(state_dict, False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        B, C, H, W = x.shape
        for i in range(len(self.local_backbone)):
            x = self.local_backbone[i](x)
            if i in self.out_indices:
                outs.append(x)
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in self.out_indices:
                outs.append(x)
        return outs


if has_mmdet:
    @det_BACKBONES.register_module()
    def mobilevig_m_feat(pretrained=True, **kwargs):
        model = MobileViG(local_blocks=[3, 3, 9],
                        local_channels=[42, 84, 224],
                        global_blocks=[3],
                        global_channels=[400],
                        dropout=0.,
                        drop_path=0.1,
                        emb_dims=768,
                        K=2,
                        distillation=True,
                        num_classes=1000,
                        out_indices=[2, 6, 16, 20],
                        pretrained='../Pretrained_Models_MobileViG/MobileViG_M_80_7.pth.tar')
        model.default_cfg = default_cfgs['mobilevig']
        return model

    @det_BACKBONES.register_module()
    def mobilevig_b_feat(pretrained=True, **kwargs):
        model = MobileViG(local_blocks=[5, 5, 15],
                        local_channels=[42, 84, 240],
                        global_blocks=[5],
                        global_channels=[464],
                        dropout=0.,
                        drop_path=0.1,
                        emb_dims=768,
                        K=2,
                        distillation=True,
                        num_classes=1000,
                        out_indices=[4, 10, 26, 32],
                        pretrained='../Pretrained_Models_MobileViG/MobileViG_B_82_6.pth.tar')
        model.default_cfg = default_cfgs['mobilevig']
        return model