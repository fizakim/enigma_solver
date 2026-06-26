"""CE approximator built as a plaintext denoiser.

The approximator estimates the true cross-entropy of a Q-Net's output against the
(unknown) plaintext, and — critically — produces a gradient that points each output
position at its *specific* true character rather than at a generic English prior.

Construction
------------
Let ``d = softmax(logits / tau)`` be the Q-Net's soft output. A denoiser ``D`` predicts,
per position, a distribution ``q = D(d)`` over the true plaintext character. With ``q``
detached, the surrogate loss

    CE_pred = -(1/T) * sum_t sum_c q_t[c] * log d_t[c]

has gradient ``(d_t - q_t) / (T * tau)`` w.r.t. the logits — the exact functional form of
the true-CE gradient ``(d_t - onehot(y_t)) / T``, with ``q_t`` standing in for the unknown
``onehot(y_t)``. When ``D`` recovers the plaintext, both the value and the gradient of
CE_pred equal true CE.

Unlike the n-gram / transformer losses (valid only at exact rotor configs, misleading on the
soft-permutation blurs the optimizer actually visits), ``D`` reads the soft output directly
and is trained to map soft blurs to the true character — so its gradient stays honest across
the whole optimization trajectory.
"""

import torch
import torch.nn as nn

from enigma_net.loss import LossFunction
from transformer.config import LMConfig
from transformer.model import CharTransformer


class PlaintextDenoiser(nn.Module):
    """Bidirectional transformer that maps a soft Q-Net output sequence to per-position
    logits over the true plaintext character.

    Reuses ``CharTransformer`` (whose ``embed`` already accepts soft distributions via
    ``x @ tok_emb.weight``) in non-causal mode so that output position ``t`` can attend to
    its own input ``d_t`` and both-side context.
    """

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = CharTransformer(cfg)

    def forward(self, d):
        """d: [B, L, n] soft distribution  ->  logits: [B, L, n]"""
        return self.transformer(d)

    @classmethod
    def from_pretrained_lm(cls, lm_ckpt_path, device="cpu", dropout=None):
        """Warm-start from a pretrained (causal) language model checkpoint.

        The weights are shape-compatible; only the attention mask differs. Inheriting the
        LM's English structure bootstraps the denoiser.
        """
        ckpt = torch.load(lm_ckpt_path, map_location="cpu")
        cfg = LMConfig.from_dict(ckpt["config"])
        cfg.causal = False
        if dropout is not None:
            cfg.dropout = dropout
        model = cls(cfg)
        model.transformer.load_state_dict(ckpt["model"])
        return model.to(device)


class CEApproximator(LossFunction):
    """Differentiable cross-entropy surrogate: ``CE_pred = CE(d, stopgrad(D(d)))``.

    Drop-in replacement for NgramLoss / TransformerLoss. Long sequences are processed in
    ``block_size`` windows (the denoiser's context length); windows are non-overlapping and
    cover every position exactly once.
    """

    requires_full_sequence = True

    def __init__(self, denoiser: PlaintextDenoiser, tau=0.5, block_size=None,
                 win_batch=256, eps=1e-9):
        super().__init__()
        self.denoiser = denoiser
        self.tau = tau
        self.block_size = block_size or denoiser.cfg.block_size
        self.win_batch = win_batch
        self.eps = eps

    def set_tau(self, tau):
        self.tau = tau

    def _run_denoiser(self, x):
        """x: [M, L, n] windows -> q: [M, L, n] predicted target distributions."""
        outs = []
        for s in range(0, x.shape[0], self.win_batch):
            logits = self.denoiser(x[s:s + self.win_batch])
            outs.append(torch.softmax(logits, dim=-1))
        return torch.cat(outs, dim=0)

    def _denoise(self, d):
        """d: [B, T, n] -> q: [B, T, n], applying the denoiser over block_size windows."""
        B, T, n = d.shape
        bs = self.block_size
        n_full = T // bs
        parts = []
        if n_full > 0:
            main = d[:, :n_full * bs, :].reshape(B * n_full, bs, n)
            q_main = self._run_denoiser(main).reshape(B, n_full * bs, n)
            parts.append(q_main)
        if T - n_full * bs > 0:
            tail = d[:, n_full * bs:, :]
            parts.append(self._run_denoiser(tail))
        return torch.cat(parts, dim=1)

    def predict_target(self, logits):
        """Return the (detached) predicted target distribution q for given logits.

        Exposed for diagnostics (recovery accuracy, gradient checks).
        """
        d = torch.softmax(logits / self.tau, dim=-1)
        with torch.no_grad():
            return self._denoise(d)

    def forward(self, logits, targets=None):
        """logits: [B, T, n]  ->  ce_estimate: [B]"""
        d = torch.softmax(logits / self.tau, dim=-1)
        # q is a fixed target: build it without autograd so no gradient flows through the
        # denoiser. The only gradient path is through log(d), giving (d - q) / (T * tau).
        with torch.no_grad():
            q = self._denoise(d)
        log_d = torch.log(d.clamp_min(self.eps))
        return -(q * log_d).sum(-1).mean(-1)


def save_ce_approximator(approximator: CEApproximator, path: str) -> None:
    torch.save({
        "denoiser": approximator.denoiser.transformer.state_dict(),
        "lm_config": approximator.denoiser.cfg.to_dict(),
        "tau": approximator.tau,
        "block_size": approximator.block_size,
    }, path)


def load_ce_approximator(path: str, device: str = "cpu") -> CEApproximator:
    ckpt = torch.load(path, map_location=device)
    cfg = LMConfig.from_dict(ckpt["lm_config"])
    cfg.causal = False
    denoiser = PlaintextDenoiser(cfg)
    denoiser.transformer.load_state_dict(ckpt["denoiser"])
    approx = CEApproximator(
        denoiser, tau=ckpt["tau"], block_size=ckpt["block_size"],
    ).to(device)
    approx.eval()
    for p in approx.parameters():
        p.requires_grad_(False)
    return approx
