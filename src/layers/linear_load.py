from torch import nn
from src.models.model_args import ModelArgs, LoraArgs
from src.layers.loralib import LoRALinear

def create_linear(in_features, out_features, bias=False, lora_args:LoraArgs=None):
    if lora_args != None:
        return LoRALinear(in_features, 
                          out_features, 
                          bias, 
                          rank=lora_args.lora_attn_dim, 
                          lora_alpha=lora_args.lora_attn_alpha, 
                          lora_dropout=lora_args.lora_dropout, 
                          fan_in_fan_out=False,
                          merge_weights=False)
    else:
        return nn.Linear(in_features, out_features, bias)

