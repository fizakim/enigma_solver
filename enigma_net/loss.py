import torch
import torch.nn as nn

class LossFunction(nn.Module):
    # When True, the loss consumes a full per-candidate sequence [B, T, n] instead of
    # flattened per-token logits (needed for sequential losses like n-gram). Per-token
    # losses such as cross-entropy leave this False.
    requires_full_sequence = False

    def forward(self, predictions, targets=None):
        raise NotImplementedError("Subclasses must implement forward")
