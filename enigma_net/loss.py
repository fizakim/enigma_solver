import torch
import torch.nn as nn

class LossFunction(nn.Module):
    requires_full_sequence = False

    def forward(self, predictions, targets=None, **kwargs):
        raise NotImplementedError("Subclasses must implement forward")
