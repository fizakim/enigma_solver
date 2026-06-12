import torch.nn as nn
from enigma_net.loss import LossFunction

class CrossEntropyLoss(LossFunction):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, predictions, targets=None):
        return self.loss_fn(predictions, targets)
