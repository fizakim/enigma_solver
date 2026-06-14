import torch
from enigma_net.loss import LossFunction

class PermutationLoss(LossFunction):
    def forward(self, predictions, targets):
        n = predictions.shape[-1]
        target_matrix = torch.zeros_like(predictions)
        target_matrix[torch.arange(n), targets] = 1.0
        return ((predictions - target_matrix) ** 2).sum() / (n * n)
