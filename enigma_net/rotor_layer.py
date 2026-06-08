import torch
import torch.nn as nn

class RotorLayer(nn.Module):
    def __init__(self, size, wiring=None):
        super().__init__()
        self.wiring = nn.Parameter(wiring if wiring is not None else torch.randn(size, size))

    def forward(self, v, position):
        return torch.roll(self.wiring @ torch.roll(v, position), -position)

    def backward_pass(self, v, position):
        return torch.roll(self.wiring.T @ torch.roll(v, position), -position)

