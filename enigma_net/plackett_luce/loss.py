import torch
from enigma_net.loss import LossFunction
from enigma_net.plackett_luce.core import Permutation

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
