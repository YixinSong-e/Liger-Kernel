import torch.nn as nn

from liger_kernel.ops.relu2 import LigerReLU2MulFunction


class LigerRELU2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(LigerReLU2MulFunction.apply(self.gate_proj(x)))
