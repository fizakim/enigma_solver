import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LMConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, n_head, T, d_head]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        # Flash/causal attention; gradient flows to x (and thus to a soft input).
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    """Decoder-only character-level language model.

    The key feature is `embed`, which accepts EITHER hard token ids (training /
    teacher forcing on real English) OR a soft per-position distribution over the
    alphabet (the q_net's decode `d`). For a soft input the token embedding is the
    *expected* embedding `d @ E`, a convex combination of the per-character
    embeddings. This single dual-path embedding is what lets the same frozen network
    serve as a differentiable unsupervised loss: gradients flow through `d @ E` back
    into `d` and hence into the q_net.
    """

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 style).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed(self, x):
        """Token embedding for hard ids [B, T] (long) or soft dist [B, T, vocab] (float).

        Soft path uses the expected embedding `x @ E`; for a one-hot `x` this is
        exactly the hard lookup, so the two paths agree on hard inputs.
        """
        if x.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            return self.tok_emb(x)
        return x @ self.tok_emb.weight

    def forward(self, x):
        """x: [B, T] long ids OR [B, T, vocab] soft dist. Returns logits [B, T, vocab]."""
        T = x.shape[1]
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        )
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.embed(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, greedy=False):
        """Autoregressively extend `idx` ([B, T] long) by `max_new_tokens` tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = probs.argmax(dim=-1, keepdim=True) if greedy else torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
