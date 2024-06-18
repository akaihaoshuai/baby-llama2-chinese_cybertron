#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
from typing import Optional, List

# from https://github.com/OptimalScale/LMFlow/tree/main
# from https://arxiv.org/abs/2403.17919
# from https://www.jiqizhixin.com/articles/2024-04-01-13

class LISA_ft():
    def __init__(
        self, 
        act_layers: int, 
        interval_steps: int, 
        model,
    ):
        self.act_layers = act_layers
        self.interval_steps = interval_steps
        self.model = model
        self.total_layers = model.n_layers
        self.act_layers_indices = []

        self.switch_active_layers()
        self.set_train_status()
     
    def switch_active_layers(self):
        # Randomly select n_layers to activate
        self.act_layers_indices = np.random.choice(range(self.total_layers), self.act_layers, replace=False)

        # layers = eval('self.' + self.model.layers_attribute)  # Re-fetch layer references
        print(f"Activating layers at indices: {self.act_layers_indices} for the next steps.", flush=True)

    def set_train_status(self):
        self.model.train()
        # Enable gradients only for the selected layers
        for idx, layer in enumerate(self.model.layers):
            if idx in self.act_layers_indices:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False
