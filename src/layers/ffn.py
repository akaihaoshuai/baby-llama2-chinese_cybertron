import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, use_bias: False, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y=self.w1(x)
        y=F.silu(y)
        z=self.w3(x)
        w=self.w2(y*z)
        x=self.dropout(w)
        return x