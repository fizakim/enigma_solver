import torch
import torch.nn as nn
from enigma_net.loss import LossFunction

class ReflectorLoss(LossFunction):
    def forward(self, model):
        R = model.reflector
        I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
        return nn.functional.mse_loss(R @ R, I)
