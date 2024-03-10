import math
import struct
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from src.models.layers.layernorm import RMSNorm
from src.models.layers.attention import Attention
from src.models.layers.ffn import FeedForward
from src.models.layers.position_code import *
from src.models.model_args import ModelArgs, LLMModelOutput
from src.models.layers.sampler import Sampler

class JerryTransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, params):
        super().__init__()
        self.n_heads = params.n_heads
        self.dim = params.dim
        self.head_dim = params.dim // params.n_heads
        self.self_attn = Attention(params, layer_idx)
        self.mlp = FeedForward(
            dim=params.dim,
            hidden_dim=4 * params.dim,
            multiple_of=params.multiple_of,
            use_bias=params.use_bias,
            dropout=params.dropout,
        )
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(params.dim, eps=params.norm_eps)
        self.post_layernorm = RMSNorm(params.dim, eps=params.norm_eps)

    def forward(self, 
                hidden_status, 
                position_ids,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]]=None,
                use_kv_cache: Optional[bool]=False,
                ):
        src_hidden_status = hidden_status
        hidden_status, past_key_value = self.self_attn.forward(self.input_layernorm(hidden_status), 
                                                               position_ids=position_ids,
                                                               attention_mask=attention_mask,
                                                               past_key_value=past_key_value,
                                                               use_kv_cache=use_kv_cache)
        hidden_status = src_hidden_status + hidden_status

        hidden_status = hidden_status + self.mlp.forward(self.post_layernorm(hidden_status))
        return hidden_status, past_key_value


class Jerry(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.is_train = params.is_train
        self.flash_attention = params.flash_attention

        self.use_neftune = params.use_neftune
        self.neftune_noise_alpha = params.neftune_noise_alpha

        # vocab_size = self.vocab_size
        vocab_size = ((params.vocab_size + 63) // 64) * 64

        self.output = nn.Linear(params.dim, vocab_size, bias=params.use_bias)
        if (params.ft_type == 'lora' or params.lora_path != '') and (params.lora_mudule == 'embedding' or params.lora_mudule == 'all'):
            from src.loralib.layers import LoRAEmbedding
            self.tok_embeddings = LoRAEmbedding(vocab_size, params.dim,
                                                r=params.lora_attn_dim,
                                                lora_alpha=params.lora_attn_alpha,
                                                merge_weights=False)
        else:
            self.tok_embeddings = nn.Embedding(vocab_size, params.dim)
        
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(params.n_layers):
            self.layers.append(JerryTransformerBlock(layer_idx, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.sampler = Sampler()
        self.attention_mask= None

        if not self.flash_attention:
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))
    
        self.cache_type = params.cache_type
        if self.cache_type == 'recent':
            from src.models.layers.short_recent_kv_cache import StartRecentKVCache
            self.kv_cache = StartRecentKVCache(
                start_size=params.cache_start_size,
                recent_size=params.cache_recent_size,
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

    # from https://github.com/neelsjain/NEFTune
    def neftune_embedding(self, tokens):
        h = self.tok_embeddings(tokens)
        dims = torch.tensor(h.size(1) * h.size(2))
        mag_norm = self.neftune_noise_alpha/torch.sqrt(dims)
        return h + torch.zeros_like(h).uniform_(-mag_norm, mag_norm)

    def forward(self, input_ids: torch.Tensor, 
                targets: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None,
                use_kv_cache: Optional[bool]=False,
                ):
        _bsz, seqlen = input_ids.shape

        past_key_value_length = 0
        if past_key_values is not None and len(past_key_values) == self.n_layers: 
            past_key_value_length = past_key_values[0][0].shape[2]

        if self.is_train and self.use_neftune:
            hidden_states = self.neftune_embedding(input_ids)
        else:
            hidden_states = self.tok_embeddings(input_ids)

        hidden_states = self.dropout(hidden_states)
        position_ids = torch.arange(seqlen+past_key_value_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)

        if not self.flash_attention and not self.flash:
            self.attention_mask = torch.full((_bsz, 1, seqlen+past_key_value_length, 
                                              seqlen+past_key_value_length), float("-inf"))
            self.attention_mask = torch.triu(self.attention_mask, diagonal=1)
        else:
            self.attention_mask = torch.ones((_bsz, seqlen+past_key_value_length), dtype=torch.long)

        new_past_key_values = []
        for idx, layer in enumerate(self.layers):
            if past_key_values is not None and len(past_key_values) == self.n_layers: 
                past_key_value = past_key_values[idx]
            else:
                past_key_value = None

            hidden_states, past_key_value = layer(hidden_states, 
                                                  position_ids=position_ids, 
                                                  attention_mask=self.attention_mask, 
                                                  past_key_value=past_key_value,
                                                  use_kv_cache=use_kv_cache)
            if use_kv_cache:
                new_past_key_values.append(past_key_value)
                
        hidden_states = self.norm(hidden_states)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(hidden_states)
            last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            ppl_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), reduction='sum')

            return LLMModelOutput(loss=last_loss, 
                                  ppl_loss=ppl_loss, 
                                  logits=logits, 
                                  hidden_states=hidden_states, 
                                  past_key_values=new_past_key_values)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(hidden_states[:, [-1], :]) # note: using list [-1] to preserve the time dim

            return LLMModelOutput(logits=logits, 
                                  hidden_states=hidden_states, 
                                  past_key_values=new_past_key_values)


    def print_params(self):
        param_dict = {pn: p for pn, p in self.tok_embeddings.named_parameters()}
        decay_params1 = [p for n, p in param_dict.items() if p.dim() >= 2 and p.requires_grad]
        num_decay_params1 = sum(p.numel() for p in decay_params1)

        param_dict = {pn: p for pn, p in self.layers[0].named_parameters()}
        decay_params2 = [p for n, p in param_dict.items() if p.dim() >= 2 and p.requires_grad]
        num_decay_params2 = sum(p.numel() for p in decay_params2)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or p.requires_grad==False]
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        params_to_update = filter(lambda p: p.requires_grad, self.parameters())
        num_params_to_update = sum(p.numel() for p in params_to_update)

        tensor_n1, tensor_n2 = len(decay_params1), len(decay_params2)

        print(f"=================models=================\n",self)
        print(f"=================models:para=================\n",self.params)
        print(f"[tok_embeddings]: num decayed parameter tensors: {tensor_n1}, with {num_decay_params1} parameters")
        print(f"[layers]: num decayed parameter tensors: {tensor_n2}*{len(self.layers)}, with {num_decay_params2}*{len(self.layers)} parameters")
        print(f"num decayed parameter tensors: {num_decay_params1+num_decay_params2*len(self.layers)} parameters")
        print(f"num non-decayed parameter tensors {num_nodecay_params} parameters")
        print(f"\nnum need-updated parameter tensors {num_params_to_update} parameters")



    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
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
    def generate(self, input_ids, eos=2, 
                 max_new_tokens : Optional[int] = 1024, 
                 temperature : Optional[float] = 1.0, 
                 top_k : Optional[int] = 10,
                 top_p : Optional[float] = 0.4,
                 use_kv_cache : Optional[bool] = True,
                 past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        
        if self.cache_type == 'recent' and use_kv_cache and past_key_values is not None:
            space_needed = input_ids.shape[1] + max_new_tokens
            past_key_values = self.kv_cache.evict_for_space(past_key_values, space_needed) # 只取需要的缓存

        # 预填充
        outputs = self(input_ids=input_ids,
                       use_kv_cache=use_kv_cache,
                       past_key_values=past_key_values,
                       )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [pred_token_idx.item()]

        # iter
        for _ in range(max_new_tokens - 1):
            # forward the model to get the logits for the index in the sequence
            outputs = self(input_ids=pred_token_idx,
                           past_key_values=past_key_values,
                           )
            past_key_values = outputs.past_key_values
            pred_token_idx=self.sampler(outputs.logits[:, -1, :], 
                                        temperature=temperature, 
                                        top_k=top_k, 
                                        top_p=top_p)
            generated_ids.append(pred_token_idx.item())
                
            if pred_token_idx==eos:
                break

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
        p = self.params
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
        print(f"wrote {filepath}")
