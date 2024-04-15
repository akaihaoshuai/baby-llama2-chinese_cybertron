import copy
from pathlib import Path

from tqdm import tqdm
from beartype import beartype
from beartype.typing import Tuple, Optional
from munch import Munch

import torch
from torch import nn

def exists(val):
    return val is not None

# Reward Model - model with a scalar head
@beartype
class RewardModel(nn.Module):
    def __init__(self, opt, model,):
        super().__init__()

        self.model = copy.deepcopy(model)
        # self.model.set_dropout(dropout)

        self.calculate_lm_loss: bool = getattr(opt, 'reward_lm_loss_factor', 0.) > 0.

        # if exists(self.reward_lora_scope):
        #     self.model.add_finetune_params(reward_lora_scope, lora_r = lora_r)

        self.dim = model.lm_head.in_features
        self.reward_head = torch.nn.Linear(self.dim, 1, bias=False)


    def forward(self, x,):
        output = self.model.forward(input_ids=x, return_dict=True, use_cache=False, output_hidden_states=True)
        hidden_states = output.hidden_states[-1]
        last_hidden_state = output.hidden_states[-1][:, -1, :]
        logits = self.reward_head(hidden_states).squeeze(-1)
     
        if self.calculate_lm_loss:
            lm_logits = self.lm_head(last_hidden_state)
            return Munch(logits=logits, lm_logits=lm_logits)
        else:
            return Munch(logits=logits)