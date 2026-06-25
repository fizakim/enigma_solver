import torch
from enigma_net.loss import LossFunction

_ELEM_BUDGET = 64_000_000

class NgramLoss(LossFunction):
    requires_full_sequence = True

    def __init__(self, log_probs, tau=1.0, t_tile=64):
        super().__init__()
        self.register_buffer("log_probs", log_probs)
        self.k = log_probs.ndim
        self.tau = tau
        self.t_tile = t_tile

    def set_tau(self, tau):
        self.tau = tau

    def _chunk_score(self, d, e0, e1):
        table_letters = "ijkl"[: self.k]
        acc = self.log_probs
        acc_sub = table_letters
        remaining = table_letters
        for m in range(self.k):
            lm = table_letters[m]
            sl = d[:, e0 - (self.k - 1) + m : e1 - (self.k - 1) + m, :]
            remaining = remaining.replace(lm, "")
            out_sub = "BT" + remaining
            acc = torch.einsum(f"BT{lm},{acc_sub}->{out_sub}", sl, acc)
            acc_sub = out_sub
        return acc.sum(dim=1)

    def forward(self, logits, targets=None):
        d = torch.softmax(logits / self.tau, dim=-1)
        B, T, _ = d.shape
        first_end = self.k - 1
        num_terms = T - first_end
        if num_terms <= 0:
            return torch.zeros(B, device=d.device, dtype=d.dtype)

        n = d.shape[-1]
        budget_tile = max(1, _ELEM_BUDGET // (B * n ** (self.k - 1)))
        tile = max(1, min(self.t_tile, budget_tile))

        score = torch.zeros(B, device=d.device, dtype=d.dtype)
        for e0 in range(first_end, T, tile):
            e1 = min(e0 + tile, T)
            score = score + self._chunk_score(d, e0, e1)

        return -score / num_terms

