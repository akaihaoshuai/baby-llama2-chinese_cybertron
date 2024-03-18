import torch
from dataclasses import dataclass
from transformers.utils.generic import ModelOutput
from typing import Optional, Tuple


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    dropout: float = 0.0
    use_bias : bool = False
    group_size_ratio : float = 0.25
    use_ssa_min_seq : int = 8192
    dtype : str = 'float16'
    model_type : str = 'Model'
    
    flash_attention : bool = False
    use_shift_short_attn : bool = False

    max_seq_len: int = 2048
    rope_scaling_type: str = 'linear'  # linear/dynamic/clex
    rope_beta: float = 10000.0
    rope_scaling_factor: float = 1.0

    is_train: bool = False
    load_in_4bit: bool = False
    
    # finutune
    ft_type : str = 'full_ft'    # full_ft/lora/qlora/dora
    lora_mudule : str = 'all'    # linear/embedding/all
    lora_attn_dim : int = 8
    lora_attn_alpha : int = 128
    lora_dropout: float = 0.0
    lora_r_dropout: float = 0.0
    lora_path: str = ''

    # netfune
    use_neftune: bool = True
    neftune_noise_alpha: float = 0.1

    # inference cache
    cache_type : str = 'recent'   # all/recent
    cache_start_size : int = 10
    cache_recent_size : int = 1024

def get_model_args(opt, train_flag):
    model_args = dict(
        dim=opt.dim,
        n_layers=opt.n_layers,
        n_heads=opt.n_heads,
        n_kv_heads=opt.n_kv_heads if opt.n_kv_heads > 0 else opt.n_heads,
        vocab_size=opt.vocab_size,#64793,
        rope_beta=opt.rope_beta,
        rope_scaling_factor=opt.rope_scaling_factor,
        rope_scaling_type=opt.rope_scaling_type,
        multiple_of=opt.multiple_of,
        max_seq_len=opt.max_seq_len,
        dropout=opt.dropout,
        use_bias=opt.use_bias,
        model_type = 'Model',
        flash_attention = False,
        use_shift_short_attn=opt.use_shift_short_attn,
        group_size_ratio=opt.group_size_ratio,
        use_ssa_min_seq=opt.use_ssa_min_seq,
        use_neftune=opt.use_neftune,
        neftune_noise_alpha=opt.neftune_noise_alpha,
        dtype=opt.dtype,
        is_train=train_flag,
        load_in_4bit=opt.load_in_4bit,

        cache_type = opt.cache_type,
        cache_start_size = opt.cache_start_size,
        cache_recent_size = opt.cache_recent_size,

        # finutune
        ft_type = opt.ft_type,
        lora_mudule = opt.lora_mudule,
        lora_attn_dim = opt.lora_attn_dim,
        lora_attn_alpha = opt.lora_attn_alpha,
        lora_dropout = opt.lora_dropout,
        lora_r_dropout = opt.lora_r_dropout,
        lora_path = '',
    )  # start with model_args from command line

    return ModelArgs(**model_args)

@dataclass
class BaseLLMModelOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
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

    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CasualLLMModelOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
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

    loss: Optional[torch.FloatTensor] = None
    ppl_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None