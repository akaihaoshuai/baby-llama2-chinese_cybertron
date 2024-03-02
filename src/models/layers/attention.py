import math
import torch
import torch.nn.functional as F
from torch import nn

from src.models.layers.position_code import *
from typing import Optional
from einops import rearrange


def repeat_kv(hidden_states: torch.Tensor, num_kv_rep_groups: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=num_kv_rep_groups). The hidden states go from (batch,
    num_kv_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if num_kv_rep_groups == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, num_kv_rep_groups, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * num_kv_rep_groups, slen, head_dim)


class Attention(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
        model_parallel_size = 1
        self.num_heads = params.n_heads // model_parallel_size
        self.num_kv_heads = self.n_kv_heads // model_parallel_size
        self.num_kv_rep_groups = self.num_heads // self.num_kv_heads
        self.head_dim = params.dim // params.n_heads
        self.wo = nn.Linear(params.n_heads * self.head_dim, params.dim, bias=params.use_bias)
        self.attn_dropout = nn.Dropout(params.dropout)
        self.resid_dropout = nn.Dropout(params.dropout)
        self.dropout = params.dropout
        self.flash_attention = params.flash_attention

        self.use_shift_short_attn = params.use_shift_short_attn
        self.group_size_ratio = params.group_size_ratio
        self.use_ssa_min_seq = params.use_ssa_min_seq

        if (params.ft_type == 'lora' or params.lora_path != '') and (params.lora_mudule == 'linear' or params.lora_mudule == 'all'):
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


    def _forward(
        self,
        x: torch.Tensor,
        freq_cos, freq_sin,
        position_ids: torch.Tensor,
        use_cache: bool = False,
        past_key_value: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        
        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, freq_cos, freq_sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": freq_cos, "cos": freq_sin}  # Specific to RoPE models
            xk, xv = past_key_value.update(xk, xv, self.layer_idx, cache_kwargs)

        # grouped multiquery attention: expand out keys and values
        # 有没有这里都没有任何区别
        xk = repeat_kv(xk, self.num_kv_rep_groups)  # (bs, num_heads, seqlen, head_dim)
        xv = repeat_kv(xv, self.num_kv_rep_groups)  # (bs, num_heads, seqlen, head_dim) 

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, 
                                                                        dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            attn_weights = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            attn_weights = attn_weights + self.mask[:, :, :seqlen, :seqlen]   # (bs, num_heads, seqlen, cache_len + seqlen)
            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(xq)
            attn_weights = self.attn_dropout(attn_weights)
            output = torch.matmul(attn_weights, xv)  # (bs, num_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        return self.resid_dropout(output)

    def forward_flashattn(
        self,
        x: torch.Tensor,
        freq_cos, freq_sin,
        position_ids: torch.Tensor,
        use_cache: bool = False,
        past_key_value: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        
        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, freq_cos, freq_sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": freq_cos, "cos": freq_sin}  # Specific to RoPE models
            xk, xv = past_key_value.update(xk, xv, self.layer_idx, cache_kwargs)

        # grouped multiquery attention: expand out keys and values
        # 有没有这里都没有任何区别
        xk = repeat_kv(xk, self.num_kv_rep_groups)  # (bs, num_heads, seqlen, head_dim)
        xv = repeat_kv(xv, self.num_kv_rep_groups)  # (bs, num_heads, seqlen, head_dim) 

        xq = xq.transpose(1, 2)  # (bs, seqlen, num_heads, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
    
        from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
        output = flash_attn_func(xq, xk, xv, self.dropout, causal=True)

        # qkv = torch.stack((xq, xk, xv), 2)
        # output = flash_attn_qkvpacked_func(qkv, self.dropout, causal=True)

        # restore time as batch dimension and concat heads
        output = output.contiguous().view(bsz, seqlen, -1)
        
        # final projection into the residual stream
        output = self.wo(output)
        return self.resid_dropout(output)


    def shift_short_attn_forward_flashattn(
        self,
        x: torch.Tensor,
        freq_cos, freq_sin,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[torch.Tensor] = None,
    ):
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    
        """Input shape: Batch x Time x Channel

        attention_mask: [bsz, q_len]
        """
        bsz, q_len, _ = x.size()

        query_states, key_states, value_states = self.wq(x), self.wk(x), self.wv(x)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, freq_cos, freq_sin, position_ids
        )

        # Past Key value support
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_kv_rep_groups)
        value_states = repeat_kv(value_states, self.num_kv_rep_groups)

        # Flash attention codes from
        # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

        # transform the data into the format required by flash attention
        qkv = torch.stack(
            [query_states, key_states, value_states], dim=2
        )  # [bsz, nh, 3, q_len, hd]
        qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

        # We have disabled _prepare_decoder_attention_mask in LlamaModel
        # the attention_mask should be the same as the key_padding_mask

        key_padding_mask = attention_mask
        nheads = qkv.shape[-2]
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(
            x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
        )
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )
        output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

        return self.wo(rearrange(output, "b s h d -> b s (h d)"))

    def shift_short_attn_forward_noflashattn(
        self,
        x: torch.Tensor,
        freq_cos, freq_sin,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[torch.Tensor] = None,
    ):
        bsz, q_len, _ = x.size()
       
        query_states, key_states, value_states = self.wq(x), self.wk(x), self.wv(x)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, freq_cos, freq_sin, position_ids
        )
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_kv_rep_groups)
        value_states = repeat_kv(value_states, self.num_kv_rep_groups)

        # shift
        def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
            qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
            qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
            return qkv

        group_size = int(q_len * self.group_size_ratio)

        if q_len % group_size > 0:
            raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_group = q_len // group_size

        query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
            raise ValueError(
                f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
            if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
                raise ValueError(
                    f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

        # shift back
        attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.wo(attn_output)
        return self.resid_dropout(attn_output)
    
    def forward(
        self,
        x: torch.Tensor,
        freq_cos, freq_sin,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[torch.Tensor] = None,
    ):
        _, seqlen, _ = x.shape
        group_size = int(seqlen * self.group_size_ratio)
        if self.use_shift_short_attn and seqlen >= self.use_ssa_min_seq and seqlen % group_size == 0:
            if self.flash_attention:
                output = self.shift_short_attn_forward_flashattn(x, freq_cos, freq_sin, position_ids, past_key_value)
            else:
                output = self.shift_short_attn_forward_noflashattn(x, freq_cos, freq_sin, position_ids, past_key_value)
        else:
            if self.flash_attention:
                output = self.forward_flashattn(x, freq_cos, freq_sin, position_ids, past_key_value)
            else:
                output = self._forward(x, freq_cos, freq_sin, position_ids, past_key_value)
        
        return output
