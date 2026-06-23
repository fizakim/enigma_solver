import torch
import torch.nn.functional as F

from enigma_net.loss import LossFunction
from .config import LMConfig
from .model import CharTransformer


def load_transformer_lm(path, device="cpu"):
    """Load a pretrained char-LM checkpoint as a frozen, eval-mode model.

    The checkpoint stores both the weights and the LMConfig (see transformer/train.py)
    so the architecture is rebuilt exactly. Mirrors `load_ngram_logprobs`.
    """
    ckpt = torch.load(path, map_location="cpu")
    cfg = LMConfig.from_dict(ckpt["config"])
    model = CharTransformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class TransformerLoss(LossFunction):
    """Unsupervised language-model loss backed by a frozen char-level transformer.

    A drop-in replacement for `NgramLoss`: it consumes a full per-candidate sequence
    of logits [B, T, n], forms the soft decode `d = softmax(logits / tau)`, and scores
    the *expected* next-character negative log-likelihood under the frozen LM:

        L = -(1 / (L-1)) * sum_t  E[ log p_LM(c_{t+1} | c_<=t) ]

    where both the context (via the expected embedding `d @ E`) and the target (the
    soft `d` itself) come from the candidate decode, so gradients flow through the
    frozen LM back into the q_net. This replaces the trigram prior with a longer-range
    learned one.

    Because T can be enormous (n**3), the sequence is scored in windows of
    `block_size` (the LM's context length); non-overlapping by default. Windows are
    forwarded through the LM in token-budgeted tiles to bound memory.
    """

    requires_full_sequence = True

    def __init__(self, lm: CharTransformer, tau=0.5, block_size=None, stride=None,
                 max_tokens=1 << 16):
        super().__init__()
        self.lm = lm
        # Keep the scorer frozen and deterministic (no dropout) regardless of how the
        # surrounding q_net toggles train/eval — loss_fn is not a child of the q_net.
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.tau = tau
        self.block_size = block_size or lm.cfg.block_size
        self.stride = stride or self.block_size
        self.max_tokens = max_tokens

    def set_tau(self, tau):
        self.tau = tau

    def _windows(self, d):
        """Split d [B, T, n] into blocks [B*W, L, n] of length L<=block_size.

        Non-overlapping by default; the tail shorter than one block is dropped (like
        the n-gram tiling). Returns (blocks, B, W).
        """
        B, T, n = d.shape
        bs = self.block_size
        if T <= bs:
            return d.unsqueeze(1).reshape(B, T, n), B, 1
        # [B, W, n, bs] -> [B, W, bs, n]
        win = d.unfold(1, bs, self.stride).permute(0, 1, 3, 2).contiguous()
        W = win.shape[1]
        return win.reshape(B * W, bs, n), B, W

    def forward(self, logits, targets=None):
        """logits: [B, T, n] -> per-candidate mean expected next-char NLL [B]."""
        B, T, n = logits.shape
        if T < 2:
            return torch.zeros(B, device=logits.device, dtype=logits.dtype)

        d = torch.softmax(logits / self.tau, dim=-1)
        blocks, B, W = self._windows(d)              # [N, L, n]
        N, L, _ = blocks.shape
        if L < 2:
            return torch.zeros(B, device=logits.device, dtype=logits.dtype)

        block_tile = max(1, self.max_tokens // L)
        parts = []
        for s in range(0, N, block_tile):
            chunk = blocks[s:s + block_tile]                       # [m, L, n]
            logp = F.log_softmax(self.lm(chunk), dim=-1)           # [m, L, n]
            # Position t predicts char t+1; target is the soft decode at t+1.
            nll = -(chunk[:, 1:, :] * logp[:, :-1, :]).sum(-1)     # [m, L-1]
            parts.append(nll.sum(dim=1))                           # [m]

        nll_per_block = torch.cat(parts) / (L - 1)                 # [N]
        return nll_per_block.view(B, W).mean(dim=1)                # [B]
