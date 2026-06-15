import torch
from enigma_net.loss import LossFunction
from enigma_net.permutation.plackett_luce.core import Permutation

class PlackettLuceLoss(LossFunction):
    def forward(self, scores, target: Permutation):
        log_A = torch.log(scores[:, target.indices] + 1e-15)
        log_suffix = torch.flip(torch.logcumsumexp(torch.flip(log_A, dims=[1]), dim=1), dims=[1])
        return (-torch.diagonal(log_A).sum() + torch.diagonal(log_suffix).sum()) / scores.shape[0]
