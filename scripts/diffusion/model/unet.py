import torch
from torch import nn
from functools import partial
from .res_block import ResnetBlock
from .attention import Attention, LinearAttention
from .helpers import Downsample, Upsample, resize_map
from .model_utils import default, divisible_by, cast_tuple
from .positional_embedding import RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        data_type = torch.float32, 
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,
        flash_attn = False,
        res_type = 'TE'
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * 2 + 1 if self_condition else channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups, type=res_type)

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features, data_type=data_type)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta, data_type=data_type)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)


        self.flash_attn = flash_attn
        FullAttention = partial(Attention, flash = flash_attn)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                self.get_attn_layer(dim=dim_in,  dim_head=layer_attn_dim_head,  heads=layer_attn_heads,  full_attn=FullAttention,  layer_full_attn=layer_full_attn),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                self.get_attn_layer(dim=dim_out,  dim_head=layer_attn_dim_head,  heads=layer_attn_heads,  full_attn=FullAttention,  layer_full_attn=layer_full_attn),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def get_attn_layer(self, dim, dim_head, heads, full_attn, layer_full_attn):
        attn_klass = full_attn if layer_full_attn else LinearAttention
        return attn_klass(dim, dim_head=dim_head, heads=heads)
    
    
    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            ill_map = x_self_cond[:, 3:4, :, :]
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            m = resize_map(ill_map, x)
            x = block1(x, t, ill_map=m)
            h.append(x)
            x = block2(x, t, ill_map=m)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)

        m = resize_map(ill_map, x)
        x = self.mid_block1(x, t, ill_map=m)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t, ill_map=m)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            m = resize_map(ill_map, x)
            x = block1(x, t, ill_map=m)
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, ill_map=m)
            x = attn(x) + x
            x = upsample(x)
            
        x = torch.cat((x, r), dim = 1)
        m = resize_map(ill_map, x)
        x = self.final_res_block(x, t, ill_map=m)
        return self.final_conv(x)
