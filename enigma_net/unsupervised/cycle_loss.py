import torch
import torch.nn as nn
from enigma_net.loss import LossFunction

class CycleLoss(LossFunction):
    def forward(self, model, inputs, positions):
        model.reset(positions)
        encrypted = []
        for v in inputs:
            encrypted.append(model(v))

        model.reset(positions)
        reconstructed = []
        for enc in encrypted:
            reconstructed.append(model(enc))

        inputs_t = torch.stack(inputs)
        recon_t = torch.stack(reconstructed)
        return nn.functional.mse_loss(recon_t, inputs_t)
