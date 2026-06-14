import torch
import torch.nn.functional as F
from enigma_net.loss import LossFunction

class PermutationLoss(LossFunction):
    def frobenius_loss(self, predictions, targets):
        n = predictions.shape[-1]
        target_matrix = torch.zeros_like(predictions)
        target_matrix[torch.arange(n, device=targets.device), targets] = 1.0
        return ((predictions - target_matrix) ** 2).sum() / (n * n)

    def row_cross_entropy_loss(self, predictions, targets, eps=1e-12):
        log_probs = torch.log(torch.clamp(predictions, min=eps))
        return F.nll_loss(log_probs, targets)

    def column_cross_entropy_loss(self, predictions, targets, eps=1e-12):
        n = predictions.shape[-1]
        inverse_targets = torch.zeros(n, dtype=torch.long, device=targets.device)
        inverse_targets[targets] = torch.arange(n, device=targets.device)
        log_probs_T = torch.log(torch.clamp(predictions.T, min=eps))
        return F.nll_loss(log_probs_T, inverse_targets)

    def forward(self, predictions, targets):
        return (self.row_cross_entropy_loss(predictions, targets) + 
                self.column_cross_entropy_loss(predictions, targets)) / 2.0
