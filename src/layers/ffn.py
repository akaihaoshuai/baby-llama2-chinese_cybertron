import torch.nn.functional as F
from torch import nn
from src.layers.activation import get_act_fn

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, use_bias: False, dropout: float, act_fn='silu'):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

        self.act_fn = get_act_fn(act_fn)

    def forward(self, x):
        y=self.w1(x)
        y=self.act_fn(y)
        z=self.w2(x)
        
        w=self.w3(y*z)
        x=self.dropout(w)
        return x
    

class OPTFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, use_bias: False, dropout: float, act_fn='silu'):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_bias)

        self.act_fn = get_act_fn(act_fn)

    def forward(self, x):
        x=self.w1(x)
        x=self.act_fn(x)
        x=self.w2(x)
        return x