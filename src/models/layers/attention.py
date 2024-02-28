import math
import torch
import torch.nn.functional as F
from torch import nn

from src.models.layers.position_code import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = params.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = params.dim // params.n_heads
        self.wo = nn.Linear(params.n_heads * self.head_dim, params.dim, bias=params.use_bias)
        self.attn_dropout = nn.Dropout(params.dropout)
        self.resid_dropout = nn.Dropout(params.dropout)
        self.dropout = params.dropout
        self.flash_attention = params.flash_attention

        if (params.ft_type == 'lora' or params.lora_path != '') and (params.lora_mudule == 'embedding' or params.lora_mudule == 'all'):
            from src.loralib.layers import LoRALinear
            self.wq = LoRALinear(
                params.dim, params.n_heads * self.head_dim, 
                use_bias=params.use_bias,
                r=params.lora_attn_dim, 
                lora_alpha=params.lora_attn_alpha, 
                lora_dropout=params.lora_dropout, 
                fan_in_fan_out=True,
                merge_weights=False
            )
            self.wk = LoRALinear(
                params.dim, self.n_kv_heads * self.head_dim, 
                use_bias=params.use_bias,
                r=params.lora_attn_dim, 
                lora_alpha=params.lora_attn_alpha, 
                lora_dropout=params.lora_dropout, 
                fan_in_fan_out=True,
                merge_weights=False
            )
            self.wv = LoRALinear(
                params.dim, self.n_kv_heads * self.head_dim, 
                use_bias=params.use_bias,
                r=params.lora_attn_dim, 
                lora_alpha=params.lora_attn_alpha, 
                lora_dropout=params.lora_dropout, 
                fan_in_fan_out=True,
                merge_weights=False
            )
        else:
            self.wq = nn.Linear(params.dim, params.n_heads * self.head_dim, bias=params.use_bias)
            self.wk = nn.Linear(params.dim, self.n_kv_heads * self.head_dim, bias=params.use_bias)
            self.wv = nn.Linear(params.dim, self.n_kv_heads * self.head_dim, bias=params.use_bias)
        
            
        if not self.flash_attention:
            # use flash attention or a manual implementation?
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            if not self.flash:
                print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
                mask = torch.full((1, 1, params.max_seq_len, params.max_seq_len), float("-inf"))
                mask = torch.triu(mask, diagonal=1)
                self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        # 有没有这里都没有任何区别
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        if self.flash_attention:
            from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
            output = flash_attn_func(xq, xk, xv, self.dropout, causal=True)

            # qkv = torch.stack((xq, xk, xv), 2)
            # output = flash_attn_qkvpacked_func(qkv, self.dropout, causal=True)

            # restore time as batch dimension and concat heads
            output = output.contiguous().view(bsz, seqlen, -1)
        else:
            # make heads into a batch dimension
            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            # flash implementation
            if self.flash:
                output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            else:
                # manual implementation
                scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
                assert hasattr(self, 'mask')
                scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                scores = self.attn_dropout(scores)
                output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

            # restore time as batch dimension and concat heads
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
