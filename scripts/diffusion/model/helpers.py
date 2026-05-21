from torch import nn
from .model_utils import default
from torch.nn import functional as F
from einops.layers.torch import Rearrange


def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )
    
def resize_map(ill_map, target_x):
    return F.interpolate(ill_map, size=target_x.shape[-2:], mode='bilinear', align_corners=False)