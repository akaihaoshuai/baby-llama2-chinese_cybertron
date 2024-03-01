#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from src.loralib.layers import *


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for name, para in model.named_parameters():
        if 'lora' not in name:
            para.requires_grad = False
        else:
            para.requires_grad = True

    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError

def merge_lora_to_save_func(model: nn.Module, path) -> None:
    model_state_dict = model.state_dict()

    # 将lora权重和原始权重融合
    for k in model_state_dict:
        if 'lora_A' in k and model_state_dict[k] is not None:
            lora_b = k.replace('lora_A', 'lora_B')
            k_weight = k.replace('lora_A', 'weight')
            model_state_dict[k_weight]+=torch.matmul(model_state_dict[lora_b], model_state_dict[k]).T

    # 将非lora部分存储下来
    no_lora = {k: model_state_dict[k] for k in model_state_dict if 'lora_' not in k}
    torch.save(no_lora, path)

def merge_lora_on_load_func(model: nn.Module, lora_state_dict) -> None:
    model_state_dict = model.state_dict()

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer) \
            or isinstance(module, LoRAEmbedding)\
            or isinstance(module, LoRALinear)\
            or isinstance(module, LoRALinearMerged)\
            or isinstance(module, LoRAConv):
            
            lora_a = name+'.lora_A'
            lora_b = name+'.lora_B'
            module.weight += torch.matmul(lora_state_dict[lora_b], lora_state_dict[lora_a]).T
            module.merged = True
            module.lora_A = None
            module.lora_B = None
            

def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
