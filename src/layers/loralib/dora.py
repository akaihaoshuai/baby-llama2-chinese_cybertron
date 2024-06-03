#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

# from https://github.com/catid/dora
# from http://arxiv.org/abs/2402.09353
# from https://zhuanlan.zhihu.com/p/682515842

# from https://github.com/microsoft/LoRA/tree/main/loralib
class DoRALayer():
    def __init__(
        self, 
        rank: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.rank = rank
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
     

class DoRALinear(nn.Linear, DoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        use_bias: bool = False,
        rank: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=use_bias)
        DoRALayer.__init__(self, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if rank > 0:
            std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, rank)))
            self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        else:
            self.lora_A = self.lora_B = None

        self.reset_parameters()  # lora权重必须初始化为0
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)     

    
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.rank > 0 and not self.merged:
            # calc_weights = self.lora_B @ self.lora_A  # LoRA
            lora = torch.matmul(self.lora_B, self.lora_A)
            adapted = T(self.weight) + lora
            column_norm = adapted.norm(p=2, dim=0, keepdim=True)
            norm_adapted = adapted / column_norm
            calc_weights = self.m * norm_adapted

            return F.linear(x, calc_weights, bias=self.bias)
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

