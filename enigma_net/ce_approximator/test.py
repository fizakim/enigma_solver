"""Evaluate a trained CE approximator (plaintext denoiser).

Reports, binned by permutation hardness (the soft-permutation continuum the optimizer
actually traverses):
  * denoiser plaintext-recovery accuracy
  * value calibration of CE_pred vs true CE (MSE / MAE / Pearson / Spearman)
  * gradient alignment: cosine(grad CE_pred, grad true_CE), compared head-to-head against
    the transformer-loss and n-gram-loss gradients on the same windows. The redesign must
    stay aligned across the soft bins where those baselines go misleading.

    python -m enigma_net.ce_approximator.test [--ckpt path]
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F

from enigma_net.ce_approximator.data_gen import generate_dataset
from enigma_net.ce_approximator.model import load_ce_approximator
from enigma_net.fourier.config import alphabet26_config
from enigma_net.ngram import NgramLoss, load_ngram_logprobs
from transformer.loss import TransformerLoss, load_transformer_lm

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CE_DIR     = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")
NGRAM_PATH  = os.path.join(_ROOT, "language", "ngram", "3grams.pth")
LM_DIR      = os.path.join(_ROOT, "models")
MODELS_DIR  = os.path.join(_CE_DIR, "models")

N_CE_BINS = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pearson(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    return float((xm * ym).sum() / (xm.norm() * ym.norm() + 1e-8))


def spearman(x, y):
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    return pearson(rx, ry)


def _grad_wrt_logits(loss_scalar_per_window, logits):
    """Return per-window gradient [B, L, n] of summed per-window loss."""
    g, = torch.autograd.grad(loss_scalar_per_window.sum(), logits, retain_graph=False)
    return g


def _cos(g, g_true):
    B = g.shape[0]
    return F.cosine_similarity(g.reshape(B, -1), g_true.reshape(B, -1), dim=1)


def per_bin_report(name, values, axis, n_bins):
    """Report `values` averaged within equal-count quantile bins of the true-CE `axis`."""
    edges = torch.quantile(axis, torch.linspace(0.0, 1.0, n_bins + 1))
    idx = torch.bucketize(axis, edges[1:-1].contiguous())
    print(f"\n{name} by true-CE level (high = sharp-wrong .. mid = soft blur .. low = solved):")
    print(f"  {'bin':>3} {'CE range':>13} {'count':>6} {'value':>9}")
    for b in range(n_bins):
        m = idx == b
        v = float(values[m].mean()) if m.any() else float("nan")
        print(f"  {b:>3} {f'{edges[b]:.2f}-{edges[b+1]:.2f}':>13} {int(m.sum()):>6} {v:>9.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    ckpt = args.ckpt or sorted(glob.glob(os.path.join(MODELS_DIR, "ce_approximator_*.pth")))[-1]
    print(f"Loading checkpoint: {ckpt}")

    config      = alphabet26_config.enigma_config
    n           = len(config.alphabet)
    char_to_idx = {c: i for i, c in enumerate(config.alphabet)}
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = "".join(ch for ch in f.read() if ch in char_to_idx)

    approx = load_ce_approximator(ckpt, device=str(device))
    tau = approx.tau
    block_size = approx.block_size

    # Baselines for the gradient-alignment comparison
    ngram_loss = NgramLoss(load_ngram_logprobs(NGRAM_PATH, n, device), tau=tau).to(device)
    lm_paths = sorted(glob.glob(os.path.join(LM_DIR, "transformer_lm_*.pth")))
    tf_loss = TransformerLoss(load_transformer_lm(lm_paths[-1], device), tau=tau)

    print("\nGenerating held-out evaluation windows...")
    X, Y, CEW = generate_dataset(
        config, corpus, char_to_idx, device,
        block_size=block_size, windows_per_candidate=4,
        n_random=100, n_traj=40, traj_snapshots=8, n_near=100, n_adv=100,
    )

    rec, ce_pred, ce_true = [], [], []
    cos_ce, cos_tf, cos_ng = [], [], []

    B = 32
    for s in range(0, len(X), B):
        xb = X[s:s + B].to(device)
        yb = Y[s:s + B].to(device)

        # recovery + value (no grad)
        with torch.no_grad():
            q = approx.predict_target(xb)
            rec.append((q.argmax(-1) == yb).float().mean(dim=1).cpu())
            ce_pred.append(approx(xb).cpu())
            ce_true.append(F.cross_entropy(
                xb.reshape(-1, n), yb.reshape(-1), reduction="none"
            ).reshape(xb.shape[0], -1).mean(1).cpu())

        # gradient alignment (needs autograd): true CE vs CE_pred / transformer / ngram
        lg = xb.detach().requires_grad_(True)
        ce_per = F.cross_entropy(lg.reshape(-1, n), yb.reshape(-1), reduction="none").reshape(lg.shape[0], -1).mean(1)
        g_true = _grad_wrt_logits(ce_per, lg)

        lg = xb.detach().requires_grad_(True)
        g_ce = _grad_wrt_logits(approx(lg), lg)
        lg = xb.detach().requires_grad_(True)
        g_tf = _grad_wrt_logits(tf_loss(lg), lg)
        lg = xb.detach().requires_grad_(True)
        g_ng = _grad_wrt_logits(ngram_loss(lg), lg)

        cos_ce.append(_cos(g_ce, g_true).cpu())
        cos_tf.append(_cos(g_tf, g_true).cpu())
        cos_ng.append(_cos(g_ng, g_true).cpu())

    rec     = torch.cat(rec)
    ce_pred = torch.cat(ce_pred)
    ce_true = torch.cat(ce_true)
    cos_ce  = torch.cat(cos_ce)
    cos_tf  = torch.cat(cos_tf)
    cos_ng  = torch.cat(cos_ng)

    print("\n=== Overall ===")
    print(f"  recovery accuracy : {rec.mean():.4f}")
    print(f"  value  MSE={F.mse_loss(ce_pred, ce_true):.4f}  MAE={(ce_pred-ce_true).abs().mean():.4f}  "
          f"Pearson={pearson(ce_pred, ce_true):.4f}  Spearman={spearman(ce_pred, ce_true):.4f}")
    print(f"  grad cosine vs true-CE :  CE_approx={cos_ce.mean():+.4f}  "
          f"transformer={cos_tf.mean():+.4f}  ngram={cos_ng.mean():+.4f}")

    per_bin_report("Recovery accuracy", rec, CEW, N_CE_BINS)
    per_bin_report("CE_approx grad cosine", cos_ce, CEW, N_CE_BINS)
    per_bin_report("transformer grad cosine", cos_tf, CEW, N_CE_BINS)
    per_bin_report("ngram grad cosine", cos_ng, CEW, N_CE_BINS)


if __name__ == "__main__":
    main()
