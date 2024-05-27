import torch.nn.functional as F
from torch import nn
from src.models.model_args import *
import numpy


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    


class MOElayers(nn.Module):
    def __init__(self, dim: int, 
                 hidden_dim: int, 
                 multiple_of: int, 
                 dropout: float,
                 num_total_experts: int = 2,
                 num_experts_per_tok: int = 1,
                 ):
        super().__init__()
        self.tp_size = 1
        self.num_total_experts = num_total_experts
        self.top_k = num_experts_per_tok

        self.gate = nn.Linear(dim, self.num_total_experts, bias=False)

        self.mlp = FeedForward(dim, hidden_dim, multiple_of, dropout)
        self.expert_indicies = numpy.array_split(range(self.num_total_experts), self.tp_size)[0].tolist()
        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of, dropout)
            if idx in self.expert_indicies else None
            for idx in range(self.num_total_experts)
        ])

        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bs, num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = None
        for expert_idx in self.expert_indicies:
            expert_layer = self.experts[expert_idx]
            expert_mask = (selected_experts == expert_idx)
            expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                 keepdim=True)

            current_hidden_states = expert_layer(hidden_states).mul_(
                expert_weights)
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        return final_hidden_states.view(bs, num_tokens, hidden_dim)