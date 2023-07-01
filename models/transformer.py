# import copy
# from typing import Optional, Any, Union, Callable
from collections import OrderedDict

import torch
import torch.nn as nn
# from torch import Tensor
import torch.nn.functional as F
# from torch.nn.modules.module import Module
# from torch.nn.modules.activation import MultiheadAttention
# from torch.nn.modules.container import ModuleList
# from torch.nn.init import xavier_uniform_
# from torch.nn.modules.dropout import Dropout
# from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, **kwargs):
        super().__init__()

        # device, detype = kwargs['device'], kwargs['dtype']
        # factory_kwargs = {'device': kwargs['device'], 'dtype': kwargs['dtype']}
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_q = LayerNorm(d_model)
        self.ln_k = LayerNorm(d_model)
        self.ln_v = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q, k, v):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None

        return self.attn(q, k, v, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, q, k, v):
        x = q + self.attention(self.ln_q(q), self.ln_k(k), self.ln_v(v))
        x = x + self.mlp(self.ln_2(x))
        return x
