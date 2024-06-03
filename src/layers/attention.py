import torch
import torch.nn.functional as F
from torch import nn
import math
from src.models.model_args import *
from src.layers.position_code.rope import *
from src.utils import *
from src.layers.linear_load import create_linear


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, q_len, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos[:, :, -q_len:, :]) + (rotate_half(q) * sin[:, :, -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape query_states and key_states to match the complex representation
    query_states_r, query_states_i = query_states.float().reshape(query_states.shape[:-1] + (-1, 2)).unbind(-1)
    key_states_r, key_states_i = key_states.float().reshape(key_states.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, query_states_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, query_states_r)

    # apply rotation using real numbers
    query_states_out_r = query_states_r * freqs_cos - query_states_i * freqs_sin
    query_states_out_i = query_states_r * freqs_sin + query_states_i * freqs_cos
    key_states_out_r = key_states_r * freqs_cos - key_states_i * freqs_sin
    key_states_out_i = key_states_r * freqs_sin + key_states_i * freqs_cos

    # flatten last two dimensions
    query_states_out = torch.stack([query_states_out_r, query_states_out_i], dim=-1).flatten(3)
    key_states_out = torch.stack([key_states_out_r, key_states_out_i], dim=-1).flatten(3)

    return query_states_out.type_as(query_states), key_states_out.type_as(key_states)

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
    def __init__(self, args: ModelArgs, 
                 lora_args: LoraArgs = None,
                 flag: str = 'train'):  # train/fft/lora/dora etc
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_dim // args.n_heads

        if lora_args is not None and not(lora_args.lora_mudule == 'all' or lora_args.lora_mudule == 'attn'):
            lora_args = None
        
        self.q_proj = create_linear(args.hidden_dim, args.n_heads    * self.head_dim, bias=args.bias, lora_args=lora_args, flag=flag)
        self.k_proj = create_linear(args.hidden_dim, self.n_kv_heads * self.head_dim, bias=args.bias, lora_args=lora_args, flag=flag)
        self.v_proj = create_linear(args.hidden_dim, self.n_kv_heads * self.head_dim, bias=args.bias, lora_args=lora_args, flag=flag)
        self.o_proj = create_linear(args.n_heads * self.head_dim, args.hidden_dim, bias=args.bias, lora_args=lora_args, flag=flag)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print_rank_0("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

        self.rope_scaling_factor = args.rope_scaling_factor
        self.max_position_embeddings = args.max_seq_len
        self.rope_beta = args.rope_beta
        self.rope_scaling_type = args.rope_scaling_type
        self._init_rope()


    # from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    def _init_rope(self):
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
                from src.layers.position_code.clex import CLEXScalingRotaryEmbedding
                self.rotary_emb = CLEXScalingRotaryEmbedding(
                    dim=self.head_dim, 
                    max_position_embeddings=self.max_position_embeddings, 
                    rope_scaling_max_factor=self.rope_scaling_factor)
            else:
                raise ValueError(f"Unknown RoPE scaling type {self.rope_scaling_type}")


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        return_qk_head_hetmaps:bool=False,
    ):
        bsz, seqlen, _ = hidden_states.shape

        # QKV
        query_states, key_states, value_states = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        query_states = query_states.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, seqlen, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states, cos, sin)

        # grouped multiquery attention: expand out keys and values
        key_states = repeat_kv(key_states, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        value_states = repeat_kv(value_states, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # flash implementation
        if self.flash:
            if return_qk_head_hetmaps:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            output = torch.nn.functional.scaled_dot_product_attention(query_states, 
                                                                      key_states, 
                                                                      value_states, 
                                                                      attn_mask=None, 
                                                                      dropout_p=self.dropout if self.training else 0.0, 
                                                                      is_causal=True)
        else:
            # manual implementation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
            
            # attn_weights = attn_weights + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(query_states)
            attn_weights = self.attn_dropout(attn_weights)
            output = torch.matmul(attn_weights, value_states)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.o_proj(output)
        output = self.resid_dropout(output)

        if return_qk_head_hetmaps:
            qk_heatmap = attn_weights.detach().cpu().numpy()
        else:
            qk_heatmap=None

        return AttentionOutput(output=output,
                               past_key_value=past_key_value,
                               qk_heatmap=qk_heatmap)