import torch.nn.functional as F
from torch import nn
import numpy
from src.models.model_args import *
from src.layers.linear_load import create_linear
from src.layers.activation import get_act_fn

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, 
                 intermediate_size: int, 
                 multiple_of: int, 
                 use_bias: False, 
                 dropout: float, 
                 act_fn='silu', 
                 lora_args: LoraArgs = None,
                 ):
        super().__init__()
        intermediate_size = int(2 * intermediate_size / 3)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.w1 = create_linear(hidden_size, intermediate_size, bias=use_bias, lora_args=lora_args)
        self.w2 = create_linear(intermediate_size, hidden_size, bias=use_bias, lora_args=lora_args)
        self.w3 = create_linear(hidden_size, intermediate_size, bias=use_bias, lora_args=lora_args)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = get_act_fn(act_fn)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    


class MOElayers(nn.Module):
    def __init__(self, hidden_size: int, 
                 intermediate_size: int, 
                 num_total_experts: int = 8,
                 num_experts_per_tok: int = 2,
                 multiple_of: int = 32, 
                 use_bias: bool = False, 
                 dropout: float = 0.1, 
                 act_fn: str ='silu', 
                 lora_args: LoraArgs = None,
                 ):
        super().__init__()
        self.tp_size = 1
        self.num_total_experts = num_total_experts
        self.top_k = num_experts_per_tok

        self.gate = create_linear(hidden_size, self.num_total_experts, bias=use_bias, lora_args=lora_args)

        # self.mlp = FeedForward(hidden_size, intermediate_size, multiple_of, dropout)
        self.expert_indicies = numpy.array_split(range(self.num_total_experts), self.tp_size)[0].tolist()
        self.experts = nn.ModuleList([
            FeedForward(hidden_size, intermediate_size, multiple_of, use_bias, dropout, act_fn, lora_args=lora_args)
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