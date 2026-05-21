import math
import torch
from torch import nn
from einops import rearrange
from .model_utils import divisible_by


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False, data_type = torch.float32):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.data_type = data_type
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered.to(self.data_type)
    
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000, data_type = torch.float32):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.data_type = data_type

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.to(self.data_type)