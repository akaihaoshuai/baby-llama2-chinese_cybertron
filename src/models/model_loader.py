from torch import nn
from typing import Type
import torch.nn as nn

from src.models.Jerry import JerryModel, JerryForCausalLM
from src.models.opt import OPTModel, OPTForCausalLM

_MODEL_REGISTRY = {
    "JerryModel": JerryModel,
    "JerryForCausalLM": JerryForCausalLM,
    "OPTModel": OPTModel,
    "OPTForCausalLM": OPTForCausalLM,
}

def _get_model_architecture(architectures) -> Type[nn.Module]:
    if architectures in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[architectures]
    
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def _get_token_architecture(architectures) -> Type[nn.Module]:
    if architectures == 'JerryModel' or architectures == 'JerryForCausalLM':
        from tokenizer_model import ChatGLMTokenizer
        return ChatGLMTokenizer, True
    elif architectures == 'OPTModel' or architectures == 'OPTForCausalLM':
        from transformers import AutoTokenizer
        return AutoTokenizer, False
    else:
        from tokenizer_model import ChatGLMTokenizer
        return ChatGLMTokenizer, True