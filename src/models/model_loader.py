from torch import nn
from typing import Type
import torch.nn as nn

from src.models.Jerry import JerryForCausalLM

_MODEL_REGISTRY = {
    "Model": JerryForCausalLM,
}

def _get_model_architecture(model_type) -> Type[nn.Module]:
    if model_type in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_type]
    raise ValueError(
        f"Model architectures {model_type} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")
