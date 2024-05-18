from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelArgs:
    dim: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    vocab_size: int = 64793  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    bias: bool = False

    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0
