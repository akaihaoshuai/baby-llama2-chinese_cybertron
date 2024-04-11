"""Custom activation functions."""
import torch
import torch.nn as nn


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[1] // 2.

    Shapes:
        x: (num_tokens, 2 * d)
        return: (num_tokens, d)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from src.layers.csrc import activation_ops
        num_tokens = x.shape[0]
        d = x.shape[1] // 2
        out = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        activation_ops.silu_and_mul(out, x)
        return out


class NewGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from src.layers.csrc import activation_ops
        num_tokens = x.shape[0]
        d = x.shape[1]
        out = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        activation_ops.gelu_new(out, x)
        return out


class FastGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from src.layers.csrc import activation_ops
        num_tokens = x.shape[0]
        d = x.shape[1]
        out = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        activation_ops.gelu_fast(out, x)
        return out


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_fast": FastGELU(), # 借鉴vllm，但是估计不支持bs>1，只能bs=1时使用
    "gelu_new": NewGELU(),   # 借鉴vllm，但是估计不支持bs>1，只能bs=1时使用
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "silu_and_mul": SiluAndMul(),  # 借鉴vllm，但是估计不支持bs>1，只能bs=1时使用
}


def get_act_fn(act_fn: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn = act_fn.lower()
    if act_fn in _ACTIVATION_REGISTRY:
        return _ACTIVATION_REGISTRY[act_fn]
    else:
        return _ACTIVATION_REGISTRY["silu"]
        # raise ValueError(f"Activation function {act_fn!r} is not supported.")
