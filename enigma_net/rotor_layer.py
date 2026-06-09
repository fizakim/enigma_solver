import torch
import torch.nn as nn
from .sinkhorn import Sinkhorn

class RotorLayer(nn.Module):
    def __init__(self, size, target_wiring=None, tau=0.1, iterations=10):
        super().__init__()
        if target_wiring is not None:
            self.register_buffer("wiring", target_wiring)
            self.logits = None
            self.sinkhorn = None
        else:
            self.logits = nn.Parameter(torch.randn(size, size))
            self.sinkhorn = Sinkhorn(tau, iterations)

    def get_wiring(self):
        if self.logits is not None:
            return self.sinkhorn(self.logits)
        return self.wiring

    def forward(self, v, position):
        w = self.get_wiring()
        return torch.roll(w @ torch.roll(v, position), -position)

    def backward_pass(self, v, position):
        w = self.get_wiring()
        return torch.roll(w.T @ torch.roll(v, position), -position)
