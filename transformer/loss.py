import torch
import torch.nn.functional as F

from enigma_net.loss import LossFunction
from .config import LMConfig
from .model import CharTransformer

def load_transformer_lm(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu")
    cfg = LMConfig.from_dict(ckpt["config"])
    model = CharTransformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model

class TransformerLoss(LossFunction):
    requires_full_sequence = True

    def __init__(self, lm: CharTransformer, tau=0.5, block_size=None, stride=None,
                 max_tokens=1 << 16):
        super().__init__()
        self.lm = lm
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
        B, T, n = d.shape
        bs = self.block_size
        if T <= bs:
            return d.unsqueeze(1).reshape(B, T, n), B, 1
        win = d.unfold(1, bs, self.stride).permute(0, 1, 3, 2).contiguous()
        W = win.shape[1]
        return win.reshape(B * W, bs, n), B, W

    def forward(self, logits, targets=None):
        B, T, n = logits.shape
        if T < 2:
            return torch.zeros(B, device=logits.device, dtype=logits.dtype)

        d = torch.softmax(logits / self.tau, dim=-1)
        blocks, B, W = self._windows(d)
        N, L, _ = blocks.shape
        if L < 2:
            return torch.zeros(B, device=logits.device, dtype=logits.dtype)

        block_tile = max(1, self.max_tokens // L)
        parts = []
        for s in range(0, N, block_tile):
            chunk = blocks[s:s + block_tile]
            logp = F.log_softmax(self.lm(chunk), dim=-1)
            nll = -(chunk[:, 1:, :] * logp[:, :-1, :]).sum(-1)
            parts.append(nll.sum(dim=1))

        nll_per_block = torch.cat(parts) / (L - 1)
        return nll_per_block.view(B, W).mean(dim=1)

