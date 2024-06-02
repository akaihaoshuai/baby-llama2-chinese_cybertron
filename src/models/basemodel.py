
from torch import nn

class BasaModel(nn.Module):
    def set_train(self):
        raise NotImplementedError
    
    def set_eval(self):
        raise NotImplementedError
