import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List
from transformers.utils.generic import ModelOutput
import numpy

@dataclass
class ModelArgs:
    architecture: str = 'Cybertron'
    hidden_dim: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    vocab_size: int = 64793  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.0
    bias: bool = False
    act_fn: str = 'silu'

    embedding_type : str = "default"  # default/Abacus

    use_moe: bool = False
    num_total_experts: int = 2
    num_experts_per_tok: int = 1

    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0

    # position code
    rope_scaling_factor: float = 1.0
    # max_position_embeddings: int = 1024  # max_seq_len
    rope_beta: float = 10000.0
    rope_scaling_type: str = 'dynamic'

    # inference cache
    cache_type : str = 'recent'   # all/recent
    cache_start_size : int = 10
    cache_recent_size : int = 1024

    linear_method : str = 'linear'  # linear

    # inference cache
    cache_type : str = 'all'   # all/recent
    cache_start_size : int = 10
    cache_recent_size : int = 1024
    

@dataclass
class LoraArgs:
    # lora
    lora_attn_dim: int = 4   # 默认0，则使用full_finetune
    lora_attn_alpha: int = 128
    lora_dropout: float = 0.0
    lora_r_dropout: float = 0.0
    lora_mudule: str = 'attn'  # embedding/attn/mlp/all


@dataclass
class AttentionOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    output: Optional[Tuple[torch.FloatTensor]] = None
    past_key_value: Optional[Tuple[torch.FloatTensor]] = None
    qk_heatmap: Optional[numpy.float16] = None



@dataclass
class TransformerBlockOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    past_key_value: Optional[Tuple[torch.FloatTensor]] = None
    qk_heatmap: Optional[numpy.float16] = None


@dataclass
class BaseLLMModelOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: Optional[Tuple[torch.FloatTensor]] = None
    last_logits: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None
    last_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    qk_heatmap_lists: Optional[List[numpy.float16]] = None
    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    z_loss: Optional[torch.FloatTensor] = None