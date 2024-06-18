import math
import struct
import inspect
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.utils import *
from src.models.model_args import *
from src.layers.attention import Attention
from src.layers.ffn import FeedForward, MOElayers
from src.ft_opt.lisa import LISA_ft
from src.layers.embedding import get_embedding
from src.layers.short_recent_kv_cache import StartRecentKVCache
from src.layers.sampler import Sampler


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full(
        (tgt_len, tgt_len),
        torch.tensor(torch.finfo(dtype).min, device=device),
        device=device,
    )

    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )



class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, 
                 args: ModelArgs, 
                 lora_args: LoraArgs = None,
                 flag: str = 'train'):  # train/fft/lora/dora etc
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.hidden_dim
        self.head_dim = args.hidden_dim // args.n_heads
        self.attention = Attention(args, lora_args, flag)

        if not args.use_moe:
            self.feed_forward = FeedForward(
                    hidden_size=args.hidden_dim,
                    intermediate_size = 4 * args.hidden_dim,
                    multiple_of=args.multiple_of,
                    use_bias=args.bias,
                    dropout=args.dropout,
                    act_fn=args.act_fn,
                    lora_args=lora_args,
                    flag=flag,
                )
        else:
            self.feed_forward = MOElayers(
                hidden_size=args.hidden_dim,
                intermediate_size=4 * args.hidden_dim,
                num_total_experts=args.num_total_experts,
                num_experts_per_tok=args.num_experts_per_tok,
                multiple_of=args.multiple_of,
                use_bias=args.bias,
                dropout=args.dropout,
                act_fn=args.act_fn,
                lora_args=lora_args,
                flag=flag,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)

    def forward(self,         
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                use_kv_cache: bool = False,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                return_qk_head_hetmaps:bool=False):
        
        attn_output = self.attention.forward(self.attention_norm(hidden_states), 
                                             attention_mask=attention_mask,
                                             position_ids=position_ids,
                                             use_kv_cache=use_kv_cache,
                                             past_key_value=past_key_value,
                                             return_qk_head_hetmaps=return_qk_head_hetmaps)
        
        residual = hidden_states + attn_output.output
        out = residual + self.feed_forward.forward(self.ffn_norm(residual))

        return TransformerBlockOutput(hidden_states=out,
                                      past_key_value=attn_output.past_key_value,
                                      qk_heatmap=attn_output.qk_heatmap)


class Cybertron(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, 
                 args: ModelArgs, 
                 lora_args: LoraArgs = None,
                 tokenizer = None,
                 flag: str = 'train'):  # train/fft/lora/dora/LISA etc
        super().__init__()
        self.args = args
        self.flag = flag
        self.lora_args = lora_args
        self.vocab_size = args.vocab_size
        self.bos_id = args.bos_id
        self.eos_id = args.eos_id
        self.pad_id = args.pad_id
        self.n_layers = args.n_layers

        self.tok_embeddings = get_embedding(args.embedding_type, args.vocab_size, args.hidden_dim, tokenizer)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args, lora_args, flag))
        self.norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.output = nn.Linear(args.hidden_dim, args.vocab_size, bias=False)

        self.sampler = Sampler()

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(args.hidden_dim // args.n_heads, args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

        self.cache_type = args.cache_type
        if self.cache_type == 'recent':
            self.kv_cache = StartRecentKVCache(
                start_size=args.cache_start_size,
                recent_size=args.cache_recent_size,
                k_seq_dim=2,  # k_seq数据在第几维
                v_seq_dim=2,
            )


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)

            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def forward(self, tokens: torch.Tensor, 
                targets: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                use_kv_cache: bool = False,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                return_qk_head_hetmaps: bool = False,
                ) -> torch.Tensor:
        batch_size, seqlen = tokens.shape

        seq_length_with_past = seqlen
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seqlen + past_key_values_length, dtype=torch.long, device=tokens.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seqlen)
        else:
            position_ids = position_ids.view(-1, seqlen).long()

        inputs_embeds = self.tok_embeddings(tokens)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=tokens.device,
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seqlen),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = self.dropout(inputs_embeds)

        qk_heatmap_lists = []
        new_past_key_values = [] if use_kv_cache else None
        for idx, layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
                        
            layer_out = layer(hidden_states, 
                              attention_mask=attention_mask,
                              position_ids=position_ids,
                              use_kv_cache=use_kv_cache,
                              past_key_value=past_key_value,
                              return_qk_head_hetmaps=return_qk_head_hetmaps)
            hidden_states = layer_out.hidden_states

            if return_qk_head_hetmaps:
                qk_heatmap_lists.append(layer_out.qk_heatmap)

            if use_kv_cache:
                new_past_key_values.append(layer_out.past_key_value)

        hidden_states = self.norm(hidden_states)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(hidden_states)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            last_logits = None
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = None
            last_logits = self.output(hidden_states[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None
        
        if not return_qk_head_hetmaps:
            qk_heatmap_lists = None

        return BaseLLMModelOutput(logits=logits,
                                  last_logits=last_logits,
                                  past_key_values=new_past_key_values,
                                  last_loss=self.last_loss,
                                  last_hidden_states=hidden_states,
                                  qk_heatmap_lists=qk_heatmap_lists)


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print_rank_0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print_rank_0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print_rank_0(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.args
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    #@torch.inference_mode()
    @torch.no_grad()
    def generate(self, input_ids, 
                 max_new_tokens=128, 
                 temperature=1.0, 
                 top_k : int = 0,
                 top_p : float = 1.0,
                 use_kv_cache: bool = True,
                 streamer=None,
                 return_qk_head_hetmaps: bool = False,
                 ):
        """
        Take a conditioning sequence of indices tokens (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in set_model_eval(model) mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        if self.cache_type == 'recent' and use_kv_cache and past_key_values is not None:
            space_needed = input_ids.shape[1] + max_new_tokens
            past_key_values = self.kv_cache.evict_for_space(past_key_values, space_needed) # 只取需要的缓存

        # 预填充
        outputs = self(tokens=input_ids,
                       use_kv_cache=use_kv_cache,
                       )
        
        past_key_values = outputs.past_key_values
        pred_token_idx=self.sampler(outputs.last_logits[-1], 
                                    temperature=temperature, 
                                    top_k=top_k, 
                                    top_p=top_p)
        
        generated_ids = [pred_token_idx.item()]

        for _ in range(max_new_tokens - 1):
            outputs = self(tokens=pred_token_idx,
                           past_key_values=past_key_values,
                           return_qk_head_hetmaps=return_qk_head_hetmaps
                           )
            
            past_key_values = outputs.past_key_values
            pred_token_idx=self.sampler(outputs.last_logits[-1], 
                                        temperature=temperature, 
                                        top_k=top_k, 
                                        top_p=top_p)
            
            # append sampled index to the running sequence and continue
            generated_ids.append(pred_token_idx.item())

            if pred_token_idx==self.eos_id:
                break
        
        if return_qk_head_hetmaps:
            return generated_ids, outputs.qk_heatmap_lists
        else:
            return generated_ids

    def export(self, filepath='model.bin'):
        """export the model weights in fp32 into .bin file to be read from C"""
        f = open(filepath, 'wb')

        def serialize(t):
            d = t.detach().cpu().view(-1).numpy().astype(np.float32)
            b = struct.pack(f'{len(d)}f', *d)
            f.write(b)

        # first write out the header
        hidden_dim = self.layers[0].feed_forward.w1.weight.shape[0]
        p = self.args
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                       n_kv_heads, p.vocab_size, p.max_seq_len)
        f.write(header)

        # next write out the embedding weights
        serialize(self.tok_embeddings.weight)

        # now all the layers
        # attention weights
        for layer in self.layers:
            serialize(layer.attention_norm.weight)
        for layer in self.layers:
            serialize(layer.attention.wq.weight)
        for layer in self.layers:
            serialize(layer.attention.wk.weight)
        for layer in self.layers:
            serialize(layer.attention.wv.weight)
        for layer in self.layers:
            serialize(layer.attention.wo.weight)
        # ffn weights
        for layer in self.layers:
            serialize(layer.ffn_norm.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w1.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w2.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w3.weight)
        # final rmsnorm
        serialize(self.norm.weight)
        # note: no need to write final classifier weights due to weight sharing
        # freqs_cis
        serialize(self.freqs_cos[:p.max_seq_len])
        serialize(self.freqs_sin[:p.max_seq_len])

        # write to binary file
        f.close()
        print_rank_0(f"wrote {filepath}")