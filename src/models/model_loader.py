from torch import nn
from typing import Type
import torch.nn as nn

from src.models.cybertron import Cybertron
# from src.models.opt import OPTModel, OPTForCausalLM

_MODEL_REGISTRY = {
    "Cybertron": Cybertron,
    # "JerryForCausalLM": JerryForCausalLM,
    # "OPTModel": OPTModel,
    # "OPTForCausalLM": OPTForCausalLM,
}

def _get_model_architecture(architectures) -> Type[nn.Module]:
    if architectures in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[architectures]
    
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def _get_tokenizer(architectures, model_path) -> Type[nn.Module]:
    if architectures == 'Cybertron' or architectures == 'Cybertron':
        from src.chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
        return ChatGLMTokenizer(model_path)
    elif architectures == 'OPTModel' or architectures == 'OPTForCausalLM':
        from transformers import AutoTokenizer
        if model_path is None:
            return None
        else:
            return AutoTokenizer.from_pretrained(model_path)
    else:
        from src.chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
        return ChatGLMTokenizer(model_path)