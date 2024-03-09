import math
import torch
import torch.nn.functional as F
from torch import nn

from src.models.layers.position_code import *
from typing import Optional, Tuple
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
    def __init__(self, params, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
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

        self.rope_beta = params.rope_beta
        self.rope_scaling_factor = params.rope_scaling_factor
        self.rope_scaling_type = params.rope_scaling_type
        self.max_position_embeddings = params.max_seq_len
        
        self.use_shift_short_attn = params.use_shift_short_attn
        self.group_size_ratio = params.group_size_ratio
        self.use_ssa_min_seq = params.use_ssa_min_seq

        if (params.ft_type == 'lora' or params.lora_path != '') and (params.lora_mudule == 'linear' or params.lora_mudule == 'all'):
            from src.loralib.layers import LoRALinear
            self.q_proj = LoRALinear(
                params.dim, params.n_heads * self.head_dim, 
                use_bias=params.use_bias,
                r=params.lora_attn_dim, 
                lora_alpha=params.lora_attn_alpha, 
                lora_dropout=params.lora_dropout, 
                fan_in_fan_out=True,
                merge_weights=False
            )
            self.k_proj = LoRALinear(
                params.dim, self.n_kv_heads * self.head_dim, 
                use_bias=params.use_bias,
                r=params.lora_attn_dim, 
                lora_alpha=params.lora_attn_alpha, 
                lora_dropout=params.lora_dropout, 
                fan_in_fan_out=True,
                merge_weights=False
            )
            self.v_proj = LoRALinear(
                params.dim, self.n_kv_heads * self.head_dim, 
                use_bias=params.use_bias,
                r=params.lora_attn_dim, 
                lora_alpha=params.lora_attn_alpha, 
                lora_dropout=params.lora_dropout, 
                fan_in_fan_out=True,
                merge_weights=False
            )
        else:
            self.q_proj = nn.Linear(params.dim, params.n_heads * self.head_dim, bias=params.use_bias)
            self.k_proj = nn.Linear(params.dim, self.n_kv_heads * self.head_dim, bias=params.use_bias)
            self.v_proj = nn.Linear(params.dim, self.n_kv_heads * self.head_dim, bias=params.use_bias)
                
        if not self.flash_attention:
            # use flash attention or a manual implementation?
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self._init_rope(params)

    # from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    def _init_rope(self, params):
        if self.rope_scaling_factor <= 1.0:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_beta,
            )
        else:
            if self.rope_scaling_type == "linear":
                self.rotary_emb = LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=self.rope_scaling_factor,
                    base=self.rope_beta,
                )
            elif self.rope_scaling_type == "dynamic":
                self.rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=self.rope_scaling_factor,
                    base=self.rope_beta,
                )
            elif self.rope_scaling_type == "clex":
                from src.models.layers.clex_position_code import CLEXScalingRotaryEmbedding
                self.rotary_emb = CLEXScalingRotaryEmbedding(
                    dim=self.head_dim, 
                    max_position_embeddings=self.max_position_embeddings, 
                    rope_scaling_max_factor=self.rope_scaling_factor)
            else:
                raise ValueError(f"Unknown RoPE scaling type {self.rope_scaling_type}")


    def _preproces_qkv(self,
                       x: torch.Tensor,
                       position_ids: torch.Tensor,
                       past_key_value: Optional[Tuple[torch.Tensor]]=None,
                       use_kv_cache: Optional[bool]=False,
        ):
        bsz, seqlen, _ = x.shape
        
        # QKV
        query_states = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        key_states_cat = key_states
        if past_key_value is not None:
            key_states_cat = torch.cat([past_key_value[0], key_states], dim=2)

        # RoPE relative positional embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states_cat, cos, sin, 
                                                        seqlen, position_ids)
        
        if past_key_value is not None:
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        new_past_key_value = None
        if use_kv_cache or past_key_value is not None:
            new_past_key_value = (key_states, value_states, cos, sin)

        # grouped multiquery attention: expand out keys and values
        key_states = repeat_kv(key_states, self.num_kv_rep_groups)  # (bs, num_heads, seqlen, head_dim)
        value_states = repeat_kv(value_states, self.num_kv_rep_groups)  # (bs, num_heads, seqlen, head_dim) 

        return (query_states, key_states, value_states), new_past_key_value
    
    def _forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]]=None,
        use_kv_cache: Optional[bool]=False,
    ):
        bsz, seqlen, _ = x.shape
        
        qkv_out, new_past_key_value = self._preproces_qkv(x, position_ids, past_key_value, use_kv_cache)
        query_states, key_states, value_states = qkv_out[0], qkv_out[1], qkv_out[2]
        
        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, 
                                                                      dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                if attention_mask[:, :, :seqlen, :seqlen].size() != attn_weights.size():
                    raise ValueError(
                        f"Attention mask should be of size {attn_weights.size()}, but is {attention_mask.size()}"
                    )
                
                attn_weights = attn_weights + attention_mask[:, :, :seqlen, :seqlen]

            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(query_states)
            attn_weights = self.attn_dropout(attn_weights)
            output = torch.matmul(attn_weights, value_states)  # (bs, num_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)

        return self.resid_dropout(output), new_past_key_value

    def forward_flashattn(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]]=None,
        use_kv_cache: Optional[bool]=False,
    ):
        bsz, seqlen, _ = x.shape
        
        qkv_out, new_past_key_value = self._preproces_qkv(x, position_ids, past_key_value, use_kv_cache)
        query_states, key_states, value_states = qkv_out[0], qkv_out[1], qkv_out[2]

        query_states = query_states.transpose(1, 2)  # (bs, seqlen, num_heads, head_dim)
        key_states   = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    
        from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
        output = flash_attn_func(query_states, key_states, value_states, self.dropout, causal=True)

        # qkv = torch.stack((query_states, key_states, value_states), 2)
        # output = flash_attn_qkvpacked_func(qkv, self.dropout, causal=True)

        # restore time as batch dimension and concat heads
        output = output.contiguous().view(bsz, seqlen, -1)
        
        # final projection into the residual stream
        output = self.wo(output)

        return self.resid_dropout(output), new_past_key_value

    # from https://github.com/dvlab-research/LongLoRA
    def shift_short_attn_forward_flashattn(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]]=None,
        use_kv_cache: Optional[bool]=False,
    ):
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    
        """Input shape: Batch x Time x Channel

        attention_mask: [bsz, seqlen]
        """
        bsz, seqlen, _ = x.size()

        qkv_out, new_past_key_value = self._preproces_qkv(x, position_ids, past_key_value, use_kv_cache)
        query_states, key_states, value_states = qkv_out[0], qkv_out[1], qkv_out[2]

        # Flash attention codes from
        # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

        # transform the data into the format required by flash attention
        qkv = torch.stack(
            [query_states, key_states, value_states], dim=2
        )  # [bsz, nh, 3, q_len, hd]
        qkv = qkv.transpose(1, 3)  # [bsz, seqlen, 3, nh, hd]

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
                rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, seqlen
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )
        output = output.reshape(bsz, seqlen, self.num_heads, self.head_dim)

        return self.wo(rearrange(output, "b s h d -> b s (h d)")), new_past_key_value
        
    
    # from https://github.com/dvlab-research/LongLoRA
    def shift_short_attn_forward_noflashattn(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]]=None,
        use_kv_cache: Optional[bool]=False,
    ):
        bsz, seqlen, _ = x.size()
       
        qkv_out, new_past_key_value = self._preproces_qkv(x, position_ids, past_key_value, use_kv_cache)
        query_states, key_states, value_states = qkv_out[0], qkv_out[1], qkv_out[2]

        # shift
        def shift(qkv, bsz, seqlen, group_size, num_heads, head_dim):
            qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
            qkv = qkv.transpose(1, 2).reshape(bsz * (seqlen // group_size), group_size, num_heads, head_dim).transpose(1, 2)
            return qkv

        group_size = int(seqlen * self.group_size_ratio)

        if seqlen % group_size > 0:
            raise ValueError("seqlen %d should be divisible by group size %d."%(seqlen, group_size))
        num_group = seqlen // group_size

        query_states = shift(query_states, bsz, seqlen, group_size, self.num_heads, self.head_dim)
        key_states = shift(key_states, bsz, seqlen, group_size, self.num_heads, self.head_dim)
        value_states = shift(value_states, bsz, seqlen, group_size, self.num_heads, self.head_dim)

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

        attn_output = attn_output.reshape(bsz, seqlen, self.num_heads, self.head_dim)

        # shift back
        attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)
        attn_output = attn_output.reshape(bsz, seqlen, -1)

        attn_output = self.wo(attn_output)

        return self.resid_dropout(attn_output), new_past_key_value
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]]=None,
        use_kv_cache: Optional[bool]=False,
    ):
        _, seqlen, _ = x.shape
        group_size = int(seqlen * self.group_size_ratio)
        if self.use_shift_short_attn and seqlen >= self.use_ssa_min_seq and seqlen % group_size == 0:
            if self.flash_attention:
                output, past_key_value = self.shift_short_attn_forward_flashattn(x, 
                                                                                position_ids, 
                                                                                attention_mask, 
                                                                                past_key_value,
                                                                                use_kv_cache)
            else:
                output, past_key_value = self.shift_short_attn_forward_noflashattn(x, 
                                                                                    position_ids, 
                                                                                    attention_mask, 
                                                                                    past_key_value,
                                                                                    use_kv_cache)
        else:
            if self.flash_attention:
                output, past_key_value = self.forward_flashattn(x,
                                                                position_ids, 
                                                                attention_mask, 
                                                                past_key_value,
                                                                use_kv_cache)
            else:
                output, past_key_value = self._forward(x,
                                                        position_ids, 
                                                        attention_mask, 
                                                        past_key_value,
                                                        use_kv_cache)
        
        return output, past_key_value
