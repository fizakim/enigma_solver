import torch
import torch.nn as nn

class LossFunction(nn.Module):
    def forward(self, predictions, targets=None):
        raise NotImplementedError("Subclasses must implement forward")
