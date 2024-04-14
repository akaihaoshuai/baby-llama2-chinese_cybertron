import math
import struct
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from src.layers.layernorm import RMSNorm
from src.layers.ffn import OPTFeedForward
from src.models.model_args import *
from src.layers.sampler import Sampler
from src.layers.short_recent_kv_cache import StartRecentKVCache
from src.utils import find_layers
from src.layers.position_code.position_code import *
from src.layers.attention import repeat_kv

class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)
    

class OPTAttention(nn.Module):
    def __init__(self, params,layer_idx,) -> None:
        super().__init__()
        self.hidden_size = params.hidden_size
        model_parallel_size = 1
        total_num_heads = params.n_heads
        assert params.n_heads % model_parallel_size == 0
        self.num_heads = total_num_heads // model_parallel_size
        self.n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
        self.num_kv_heads = self.n_kv_heads // model_parallel_size
        self.num_kv_rep_groups = self.num_heads // self.num_kv_heads
        self.head_dim = params.hidden_size // total_num_heads
        
        self.scaling = self.head_dim**-0.5
        self.rope_beta = params.rope_beta
        self.rope_scaling_factor = params.rope_scaling_factor
        self.rope_scaling_type = params.rope_scaling_type
        self.max_position_embeddings = params.max_seq_len
        self.flash_attention = params.flash_attention

        if not self.flash_attention:
            # use flash attention or a manual implementation?
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.dropout = params.dropout

        self.k_proj = nn.Linear(params.hidden_size, params.hidden_size, bias=params.use_bias)
        self.v_proj = nn.Linear(params.hidden_size, params.hidden_size, bias=params.use_bias)
        self.q_proj = nn.Linear(params.hidden_size, params.hidden_size, bias=params.use_bias)
        self.out_proj = nn.Linear(params.hidden_size, params.hidden_size, bias=params.use_bias)
        self._init_rope(params)

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
                from src.layers.position_code.clex import CLEXScalingRotaryEmbedding
                self.rotary_emb = CLEXScalingRotaryEmbedding(
                    dim=self.head_dim, 
                    max_position_embeddings=self.max_position_embeddings, 
                    rope_scaling_max_factor=self.rope_scaling_factor)
            else:
                raise ValueError(f"Unknown RoPE scaling type {self.rope_scaling_type}")
            

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]]=None,
        use_kv_cache: Optional[bool]=False,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        query_states = self.q_proj(x) * self.scaling
        key_states   = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = query_states.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

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
        output = self.out_proj(output)

        return output, new_past_key_value
    

class OPTTransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, params):
        super().__init__()
        self.n_heads = params.n_heads
        self.hidden_size = params.hidden_size
        self.head_dim = params.hidden_size // params.n_heads
        self.self_attn = OPTAttention(params, layer_idx)
        self.do_layer_norm_before = params.do_layer_norm_before

        self.self_attn_layer_norm = nn.LayerNorm(params.hidden_size)
        self.final_layer_norm = nn.LayerNorm(params.hidden_size)

        self.mlp = OPTFeedForward(
            dim=params.hidden_size,
            hidden_dim=4 * params.hidden_size,
            multiple_of=params.multiple_of,
            use_bias=params.use_bias,
            dropout=params.dropout,
            act_fn=params.act_fn,
        )
        self.layer_idx = layer_idx

    def forward(self, 
                hidden_status, 
                position_ids,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]]=None,
                use_kv_cache: Optional[bool]=False,
                ):
        src_hidden_status = hidden_status
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_status)
        hidden_status, past_key_value = self.self_attn.forward(hidden_states, 
                                                               position_ids=position_ids,
                                                               attention_mask=attention_mask,
                                                               past_key_value=past_key_value,
                                                               use_kv_cache=use_kv_cache)
        hidden_status = src_hidden_status + hidden_status

        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_status = hidden_status + self.mlp.forward(hidden_status)
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_status, past_key_value


class OPTDecoder(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params
        self.padding_idx = params.pad_token_id
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.is_train = params.is_train
        self.flash_attention = params.flash_attention
        self.max_seq_len = params.max_seq_len
        self.hidden_size = params.hidden_size

        self.use_neftune = params.use_neftune
        self.neftune_noise_alpha = params.neftune_noise_alpha

        if (params.ft_type == 'lora' or params.lora_path != '') \
            and (params.lora_mudule == 'embedding' or params.lora_mudule == 'all'):
            from src.loralib.layers import LoRAEmbedding
            self.embed_tokens = LoRAEmbedding(params.vocab_size, params.word_embed_proj_dim,
                                                r=params.lora_attn_dim,
                                                lora_alpha=params.lora_attn_alpha,
                                                merge_weights=False)
        else:
            self.embed_tokens = nn.Embedding(params.vocab_size, params.word_embed_proj_dim)

        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            params.max_seq_len, params.hidden_size)

        # Project out & in will be replicated if they exist.
        if params.word_embed_proj_dim != params.hidden_size:
            self.project_out = nn.Linear(params.hidden_size,
                                         params.word_embed_proj_dim,
                                         bias=False)
        else:
            self.project_out = None

        if params.word_embed_proj_dim != params.hidden_size:
            self.project_in = nn.Linear(params.word_embed_proj_dim,
                                        params.hidden_size,
                                        bias=False)
        else:
            self.project_in = None

        if params.do_layer_norm_before:
            self.final_layer_norm = nn.LayerNorm(params.hidden_size)

        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_idx in range(params.n_layers):
            self.layers.append(OPTTransformerBlock(layer_idx, params))

        self.attention_mask= None

        if not self.flash_attention:
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))
    

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # from https://github.com/neelsjain/NEFTune
    def neftune_embedding(self, tokens):
        h = self.embed_tokens(tokens)
        dims = torch.tensor(h.size(1) * h.size(2))
        mag_norm = self.neftune_noise_alpha/torch.sqrt(dims)
        return h + torch.zeros_like(h).uniform_(-mag_norm, mag_norm)

    def forward(self, input_ids: torch.Tensor, 
                past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None,
                use_kv_cache: Optional[bool]=False,
                ):
        _bsz, seqlen = input_ids.shape
        
        past_key_value_length = 0
        if past_key_values is not None and len(past_key_values) == self.n_layers: 
            past_key_value_length = past_key_values[0][0].shape[2]

        if self.is_train and self.use_neftune:
            inputs_embeds = self.neftune_embedding(input_ids)
        else:
            inputs_embeds = self.embed_tokens(input_ids)

        position_ids = torch.arange(seqlen+past_key_value_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)

        pos_embeds = self.embed_positions(position_ids)
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds[:, -seqlen:, :]

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
                
        hidden_states = self.final_layer_norm(hidden_states)

        return BaseLLMModelOutput(hidden_states=hidden_states, 
                                  past_key_values=new_past_key_values)


    
class OPTModel(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.decoder = OPTDecoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None,
        use_kv_cache: Optional[bool]=False,
    ) -> torch.Tensor:
        return self.decoder(input_ids, past_key_values, use_kv_cache)
    


class OPTForCausalLM(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        # params.vocab_size = ((params.vocab_size + 63) // 64) * 64

        self.model = OPTModel(params)
        self.sampler = Sampler()

        # share the unembedding parameters with the embedding parameters
        self.lm_head = nn.Linear(params.word_embed_proj_dim, params.vocab_size, bias=params.use_bias)
        self.lm_head.weight = self.model.decoder.embed_tokens.weight # https://paperswithcode.com/method/weight-tying

        self.cache_type = params.cache_type
        self.kv_cache = None
        if self.cache_type == 'recent':
            self.kv_cache = StartRecentKVCache(
                start_size=params.cache_start_size,
                recent_size=params.cache_recent_size,
                k_seq_dim=2,  # k_seq数据在第几维
                v_seq_dim=2,
            )
        
        if -1 != params.load_in_lowbit:
            if 'gptq' == params.load_in_quant_type:
                from src.quant.share import gptq_make_quant_linear
                layers = find_layers(self)
                for name in ['output']:
                    if name in layers:
                        del layers[name]
                gptq_make_quant_linear(self, layers, params.load_in_lowbit, params.load_in_lowbit_groupsize)
            elif 'llm_int8' == params.load_in_quant_type:
                from transformers.utils import is_bitsandbytes_available
                if is_bitsandbytes_available():
                    layers = find_layers(self)
                    for name in ['output']:
                        if name in layers:
                            del layers[name]
                    from src.quant.share import bnb_make_quant_linear
                    bnb_make_quant_linear(self, layers, params.load_in_lowbit, params.load_in_lowbit_groupsize)
            elif 'awq' == params.load_in_quant_type:
                layers = find_layers(self)
                for name in ['output']:
                    if name in layers:
                        del layers[name]
                from src.quant.share import awq_make_quant_linear
                awq_make_quant_linear(self, layers, params.load_in_lowbit, params.load_in_lowbit_groupsize)
            elif 'bitnet' == params.load_in_quant_type:
                layers = find_layers(self)
                for name in ['output']:
                    if name in layers:
                        del layers[name]
                from src.quant.share import bitnet_make_quant_linear
                bitnet_make_quant_linear(self, layers, params.load_in_lowbit, params.load_in_lowbit_groupsize)


    def forward(self, input_ids: torch.Tensor, 
                targets: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None,
                use_kv_cache: Optional[bool]=False,
                ):
        
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_kv_cache=use_kv_cache,
        )

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(outputs.hidden_states)
            last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            ppl_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), reduction='sum')

            return CasualLLMModelOutput(loss=last_loss, 
                                        ppl_loss=ppl_loss, 
                                        logits=logits, 
                                        hidden_states=outputs.hidden_states, 
                                        past_key_values=outputs.past_key_values)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.lm_head(outputs.hidden_states[:, [-1], :]) # note: using list [-1] to preserve the time dim

            return CasualLLMModelOutput(logits=logits, 
                                        hidden_states=outputs.hidden_states, 
                                        past_key_values=outputs.past_key_values)
    
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
        outputs = self.model(input_ids=input_ids,
                            use_kv_cache=use_kv_cache,
                            past_key_values=past_key_values,
                            )
        logits = self.lm_head(outputs.hidden_states[:, -1, :]) # note: using list [-1] to preserve the time dim
        past_key_values = outputs.past_key_values
        pred_token_idx = logits.argmax(dim=-1).unsqueeze(1)
        generated_ids = [pred_token_idx.item()]

        # iter
        for _ in range(max_new_tokens - 1):
            # forward the model to get the logits for the index in the sequence
            outputs = self.model(input_ids=pred_token_idx,
                                 past_key_values=past_key_values,
                                 )
            past_key_values = outputs.past_key_values
            logits = self.lm_head(outputs.hidden_states[:, -1, :]) # note: using list [-1] to preserve the time dim
            pred_token_idx=self.sampler(logits, 
                                        temperature=temperature, 
                                        top_k=top_k, 
                                        top_p=top_p)
            generated_ids.append(pred_token_idx.item())
                
            if pred_token_idx==eos:
                break

        return generated_ids
    
    def print_params(self):
        param_dict = {pn: p for pn, p in self.model.tok_embeddings.named_parameters()}
        decay_params1 = [p for n, p in param_dict.items() if p.dim() >= 2 and p.requires_grad]
        num_decay_params1 = sum(p.numel() for p in decay_params1)

        param_dict = {pn: p for pn, p in self.model.layers[0].named_parameters()}
        decay_params2 = [p for n, p in param_dict.items() if p.dim() >= 2 and p.requires_grad]
        num_decay_params2 = sum(p.numel() for p in decay_params2)

        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or p.requires_grad==False]
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        params_to_update = filter(lambda p: p.requires_grad, self.parameters())
        num_params_to_update = sum(p.numel() for p in params_to_update)

        tensor_n1, tensor_n2 = len(decay_params1), len(decay_params2)

        print(f"=================models=================\n",self)
        print(f"=================models:para=================\n",self.model.params)
        print(f"[tok_embeddings]: num decayed parameter tensors: {tensor_n1}, with {num_decay_params1} parameters")
        print(f"[layers]: num decayed parameter tensors: {tensor_n2}*{len(self.model.layers)}, \
              with {num_decay_params2}*{len(self.model.layers)} parameters")
        print(f"num decayed parameter tensors: \
              {num_decay_params1+num_decay_params2*len(self.model.layers)} parameters")
        print(f"num non-decayed parameter tensors {num_nodecay_params} parameters")
        print(f"\nnum need-updated parameter tensors {num_params_to_update} parameters")
