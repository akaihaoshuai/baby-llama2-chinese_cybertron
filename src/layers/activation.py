"""Custom activation functions."""
import torch
import torch.nn as nn



_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    # "gelu_fast": FastGELU(), # 借鉴vllm，但是估计不支持bs>1，只能bs=1时使用
    # "gelu_new": NewGELU(),   # 借鉴vllm，但是估计不支持bs>1，只能bs=1时使用
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    # "silu_and_mul": SiluAndMul(),  # 借鉴vllm，但是估计不支持bs>1，只能bs=1时使用
}


def get_act_fn(act_fn: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn = act_fn.lower()
    if act_fn in _ACTIVATION_REGISTRY:
        return _ACTIVATION_REGISTRY[act_fn]
    else:
        return _ACTIVATION_REGISTRY["silu"]
        # raise ValueError(f"Activation function {act_fn!r} is not supported.")