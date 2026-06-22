import torch

from enigma_net.loss import LossFunction

# Element budget for the largest n-gram contraction intermediate ([B, tile, n^(k-1)]).
# ~64M floats ≈ 256 MB; the tile is shrunk to respect this regardless of t_tile.
_ELEM_BUDGET = 64_000_000


class NgramLoss(LossFunction):
    """Unsupervised n-gram language-model loss for the continuous solver.

    The model emits per-step logits; softmax(logits / tau) gives a soft decoded
    distribution d_t over letters at each position. This loss is the mean negative
    *expected* n-gram log-probability of the decoded stream:

        L = -(1 / (T - k + 1)) * sum_t  E[ logP(d_{t-k+1}, ..., d_t) ]

    where the expectation factorizes over the independent per-position distributions.
    It needs no targets — only the English-language prior baked into `log_probs`.

    Because an n-gram couples k consecutive positions, this loss consumes a full
    per-candidate sequence [B, T, n] (unlike the per-token-independent cross-entropy).
    The trainer flags this via `requires_full_sequence`.
    """

    requires_full_sequence = True

    def __init__(self, log_probs, tau=1.0, t_tile=64):
        super().__init__()
        self.register_buffer("log_probs", log_probs)
        self.k = log_probs.ndim
        self.tau = tau
        self.t_tile = t_tile

    def set_tau(self, tau):
        """Adjust the softmax temperature (e.g. for annealing). Unused by default."""
        self.tau = tau

    def _chunk_score(self, d, e0, e1):
        """Sum of expected n-gram log-probs for n-grams *ending* in [e0, e1).

        d: [B, T, n] soft distributions. Returns [B] (summed over the chunk).
        Indices are windowed with proper offsets, so no n-gram is dropped at a
        chunk boundary. Peak memory is the [B, chunk, n, n] first contraction.
        """
        table_letters = "ijkl"[: self.k]
        acc = self.log_probs              # subscripts: table_letters (no batch dims yet)
        acc_sub = table_letters
        remaining = table_letters
        for m in range(self.k):
            lm = table_letters[m]
            # The m-th character of an n-gram ending at t sits at position t-(k-1)+m.
            sl = d[:, e0 - (self.k - 1) + m : e1 - (self.k - 1) + m, :]  # [B, chunk, n]
            remaining = remaining.replace(lm, "")
            out_sub = "BT" + remaining
            acc = torch.einsum(f"BT{lm},{acc_sub}->{out_sub}", sl, acc)
            acc_sub = out_sub
        return acc.sum(dim=1)             # [B]

    def forward(self, logits, targets=None):
        """logits: [B, T, n] -> per-candidate mean negative n-gram log-prob [B]."""
        d = torch.softmax(logits / self.tau, dim=-1)
        B, T, _ = d.shape
        first_end = self.k - 1           # first valid ending position
        num_terms = T - first_end
        if num_terms <= 0:
            return torch.zeros(B, device=d.device, dtype=d.dtype)

        # The first staged contraction materializes a [B, tile, n^(k-1)] intermediate.
        # Cap the tile so that stays within a fixed element budget (important for k>=4,
        # where n^(k-1) is large), independent of the configured t_tile.
        n = d.shape[-1]
        budget_tile = max(1, _ELEM_BUDGET // (B * n ** (self.k - 1)))
        tile = max(1, min(self.t_tile, budget_tile))

        score = torch.zeros(B, device=d.device, dtype=d.dtype)
        for e0 in range(first_end, T, tile):
            e1 = min(e0 + tile, T)
            score = score + self._chunk_score(d, e0, e1)

        return -score / num_terms
