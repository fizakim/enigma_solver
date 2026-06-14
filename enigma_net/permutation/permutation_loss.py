import torch
import torch.nn.functional as F
from enigma_net.loss import LossFunction
from enigma_net.permutation.permutation_core import Permutation

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


class PlackettLuceLoss(LossFunction):
    def forward(self, scores, target: Permutation):
        n = scores.shape[0]
        sigma = target.indices
        nll = 0.0
        for i in range(n):
            nll -= torch.log(scores[i, sigma[i]] + 1e-12)
            mask = torch.ones(n, device=scores.device)
            mask[sigma[:i]] = 0.0
            nll += torch.log(torch.sum(scores[i] * mask) + 1e-12)
        return nll / n

