import torch
from enigma_net.loss import LossFunction
from enigma_net.permutation.plackett_luce.core import Permutation

class PlackettLuceLoss(LossFunction):
    def forward(self, scores, target: Permutation):
        n = scores.shape[0]
        sigma = target.indices
        A = scores[:, sigma]
        suffix_sums = torch.flip(torch.cumsum(torch.flip(A, dims=[1]), dim=1), dims=[1])
        denom = torch.diagonal(suffix_sums)
        num = torch.diagonal(A)
        nll = -torch.log(num + 1e-12).sum() + torch.log(denom + 1e-12).sum()
        return nll / n
