import math
import struct
import inspect
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.model_args import *
from src.layers.attention import Attention
from src.layers.ffn import FeedForward


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



class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, return_qk_head_hetmaps:bool=False):
        attn_output = self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, return_qk_head_hetmaps)
        h = x + attn_output.output
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return TransformerBlockOutput(hidden_states=out,
                                      past_key_value=None,
                                      qk_heatmap=attn_output.qk_heatmap)


class Cybertron(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.bos_id = params.bos_id
        self.eos_id = params.eos_id
        self.pad_id = params.pad_id
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, 
                targets: Optional[torch.Tensor] = None,
                return_qk_head_hetmaps: bool = False,
                ) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        qk_heatmap_lists = []
        for layer in self.layers:
            layer_out = layer(h, freqs_cos, freqs_sin, return_qk_head_hetmaps)
            h = layer_out.hidden_states

            if return_qk_head_hetmaps:
                qk_heatmap_lists.append(layer_out.qk_heatmap)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            last_logits = None
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = None
            last_logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None
        
        if not return_qk_head_hetmaps:
            qk_heatmap_lists = None

        return BaseLLMModelOutput(logits=logits,
                                  last_logits=last_logits,
                                  past_key_values=None,
                                  last_loss=self.last_loss,
                                  last_hidden_states=h,
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
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

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
    def generate(self, input_ids, 
                 max_new_tokens=128, 
                 temperature=1.0, 
                 top_k=None,
                 attention_mask=None,
                 position_ids=None,
                 use_cache=None,
                 streamer=None,
                 return_qk_head_hetmaps: bool = False,
                 ):
        """
        Take a conditioning sequence of indices tokens (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            token_cond = input_ids if input_ids.size(1) <= self.params.max_seq_len else input_ids[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            outputs = self(token_cond, return_qk_head_hetmaps=return_qk_head_hetmaps)

            logits = outputs.last_logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, token_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                token_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, token_next), dim=1)
            if token_next==self.eos_id:
                break
        
        if return_qk_head_hetmaps:
            return input_ids, outputs.qk_heatmap_lists
        else:
            return input_ids

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