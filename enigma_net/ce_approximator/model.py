from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from enigma_net.loss import LossFunction
from transformer.config import LMConfig
from transformer.model import CharTransformer


@dataclass
class DenoiserFeatures:
    use_cipher: bool = False
    use_positions: bool = True
    use_state: bool = True
    use_lm_prior: bool = False
    pos_dropout: float = 0.3
    num_rotors: int = 3
    n: int = 26


class PlaintextDenoiser(nn.Module):
    def __init__(self, cfg: LMConfig, feats: DenoiserFeatures = None):
        super().__init__()
        self.cfg = cfg
        self.feats = feats or DenoiserFeatures()
        self.transformer = CharTransformer(cfg)

        d_model = cfg.d_model
        if self.feats.use_cipher:
            self.cipher_emb = nn.Embedding(self.feats.n, d_model)
            nn.init.zeros_(self.cipher_emb.weight)
        if self.feats.use_positions:
            self.rotor_pos_emb = nn.ModuleList(
                [nn.Embedding(self.feats.n, d_model) for _ in range(self.feats.num_rotors)]
            )
            for emb in self.rotor_pos_emb:
                nn.init.zeros_(emb.weight)
            self.pos_drop = nn.Dropout(self.feats.pos_dropout)
        if self.feats.use_lm_prior:
            self.prior_proj = nn.Linear(self.feats.n, d_model)
            nn.init.zeros_(self.prior_proj.weight)
            nn.init.zeros_(self.prior_proj.bias)
        if self.feats.use_state:
            cond_dim = self.feats.num_rotors + 1
            self.film = nn.Sequential(
                nn.Linear(cond_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2 * d_model),
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

    @staticmethod
    def _entropy(d, eps=1e-9):
        return -(d * d.clamp_min(eps).log()).sum(-1).mean(-1, keepdim=True)

    def forward(self, d, cipher=None, positions=None, qnet_state=None, prior=None):
        h = self.transformer.embed(d)
        if self.feats.use_cipher and cipher is not None:
            h = h + self.cipher_emb(cipher)
        if self.feats.use_positions and positions is not None:
            pos_h = sum(emb(positions[..., r]) for r, emb in enumerate(self.rotor_pos_emb))
            h = h + self.pos_drop(pos_h)
        if self.feats.use_lm_prior and prior is not None:
            h = h + self.prior_proj(prior)
        if self.feats.use_state and qnet_state is not None:
            cond = torch.cat([qnet_state, self._entropy(d)], dim=-1)
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return self.transformer.forward_from_embeddings(h)

    @classmethod
    def from_pretrained_lm(cls, lm_ckpt_path, feats: DenoiserFeatures = None,
                           device="cpu", dropout=None, block_size=None):
        ckpt = torch.load(lm_ckpt_path, map_location="cpu")
        cfg = LMConfig.from_dict(ckpt["config"])
        cfg.causal = False
        if dropout is not None:
            cfg.dropout = dropout
        orig_bs = cfg.block_size
        if block_size is not None and block_size != orig_bs:
            cfg.block_size = block_size
        model = cls(cfg, feats)
        lm_state = dict(ckpt["model"])
        if cfg.block_size != orig_bs:
            old_pos = lm_state.pop("pos_emb.weight")
            model.transformer.load_state_dict(lm_state, strict=False)
            with torch.no_grad():
                keep = min(orig_bs, cfg.block_size)
                model.transformer.pos_emb.weight[:keep].copy_(old_pos[:keep])
        else:
            model.transformer.load_state_dict(lm_state)
        return model.to(device)


class CEApproximator(LossFunction):
    requires_full_sequence = True

    def __init__(self, denoiser: PlaintextDenoiser, tau=0.5, block_size=None,
                 win_batch=256, eps=1e-9, prior_lm: CharTransformer = None):
        super().__init__()
        self.denoiser = denoiser
        self.tau = tau
        self.block_size = block_size or denoiser.cfg.block_size
        self.win_batch = win_batch
        self.eps = eps
        self.prior_lm = prior_lm

    def set_tau(self, tau):
        self.tau = tau

    def _window_prior(self, d):
        if not (self.denoiser.feats.use_lm_prior and self.prior_lm is not None):
            return None
        with torch.no_grad():
            return torch.softmax(self.prior_lm(d), dim=-1)

    def _run_denoiser(self, d, cipher, positions, qnet_state, prior):
        outs = []
        for s in range(0, d.shape[0], self.win_batch):
            sl = slice(s, s + self.win_batch)
            logits = self.denoiser(
                d[sl],
                cipher[sl] if cipher is not None else None,
                positions[sl] if positions is not None else None,
                qnet_state[sl] if qnet_state is not None else None,
                prior[sl] if prior is not None else None,
            )
            outs.append(torch.softmax(logits, dim=-1))
        return torch.cat(outs, dim=0)

    def _denoise(self, d, cipher=None, positions=None, qnet_state=None):
        B, T, n = d.shape
        bs = self.block_size
        n_full = T // bs
        parts = []
        if n_full > 0:
            L = n_full * bs
            md = d[:, :L, :].reshape(B * n_full, bs, n)
            mc = cipher[:, :L].reshape(B * n_full, bs) if cipher is not None else None
            mp = positions[:, :L, :].reshape(B * n_full, bs, positions.shape[-1]) if positions is not None else None
            ms = qnet_state.repeat_interleave(n_full, dim=0) if qnet_state is not None else None
            q = self._run_denoiser(md, mc, mp, ms, self._window_prior(md)).reshape(B, L, n)
            parts.append(q)
        if T - n_full * bs > 0:
            o = n_full * bs
            td = d[:, o:, :]
            tc = cipher[:, o:] if cipher is not None else None
            tp = positions[:, o:, :] if positions is not None else None
            parts.append(self._run_denoiser(td, tc, tp, qnet_state, self._window_prior(td)))
        return torch.cat(parts, dim=1)

    def predict_target(self, logits, cipher=None, positions=None, qnet_state=None):
        d = torch.softmax(logits / self.tau, dim=-1)
        with torch.no_grad():
            return self._denoise(d, cipher, positions, qnet_state)

    def loss_with_target(self, logits, q):
        d = torch.softmax(logits / self.tau, dim=-1)
        log_d = torch.log(d.clamp_min(self.eps))
        return -(q * log_d).sum(-1).mean(-1)

    def forward(self, logits, targets=None, cipher=None, positions=None, qnet_state=None):
        d = torch.softmax(logits / self.tau, dim=-1)
        with torch.no_grad():
            q = self._denoise(d, cipher, positions, qnet_state)
        log_d = torch.log(d.clamp_min(self.eps))
        return -(q * log_d).sum(-1).mean(-1)


def save_ce_approximator(approximator: CEApproximator, path: str) -> None:
    torch.save({
        "denoiser": approximator.denoiser.state_dict(),
        "lm_config": approximator.denoiser.cfg.to_dict(),
        "feats": asdict(approximator.denoiser.feats),
        "tau": approximator.tau,
        "block_size": approximator.block_size,
    }, path)


def load_ce_approximator(path: str, device: str = "cpu") -> CEApproximator:
    ckpt = torch.load(path, map_location=device)
    cfg = LMConfig.from_dict(ckpt["lm_config"])
    cfg.causal = False
    feats = DenoiserFeatures(**ckpt["feats"])
    denoiser = PlaintextDenoiser(cfg, feats)
    denoiser.load_state_dict(ckpt["denoiser"])
    approx = CEApproximator(
        denoiser, tau=ckpt["tau"], block_size=ckpt["block_size"],
    ).to(device)
    approx.eval()
    for p in approx.parameters():
        p.requires_grad_(False)
    return approx
