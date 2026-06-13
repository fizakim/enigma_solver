import torch
import torch.nn as nn
from enigma_net.loss import LossFunction

class NgramLoss(LossFunction):
    def __init__(self, ngram_counts, eps=1e-8):
        super().__init__()
        self.eps = eps
        ref = ngram_counts.float() + eps
        ref = ref / ref.sum()
        self.register_buffer("ref", ref)
        self.n = ngram_counts.dim()

    def forward(self, predictions, targets=None):
        seq_len, K = predictions.shape
        if seq_len < self.n:
            return torch.tensor(0.0, requires_grad=True)

        windows = [predictions[i:seq_len - self.n + 1 + i] for i in range(self.n)]

        letters = "abcdefghijklmnopqrstuvwxyz"
        input_subs = ",".join(f"n{letters[i]}" for i in range(self.n))
        output_sub = "".join(letters[i] for i in range(self.n))
        einsum_str = f"{input_subs}->{output_sub}"

        soft_counts = torch.einsum(einsum_str, *windows)
        pred_dist = (soft_counts + self.eps) / (soft_counts + self.eps).sum()

        kl = (self.ref * (self.ref.log() - pred_dist.log())).sum()
        return kl
