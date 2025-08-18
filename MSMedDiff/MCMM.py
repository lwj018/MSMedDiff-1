import torch
from einops import rearrange
from torch import nn

import torch.nn.functional as F
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MDWC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        res = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return res * attn


class MCMM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.norm=LayerNorm(dim)

        self.mfr = nn.ModuleList(
            [MDWC(chunk_dim) for i in range(self.n_levels)])

        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        no_norm=x
        x=self.norm(x)
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out+no_norm

