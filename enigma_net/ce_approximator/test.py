import argparse
import glob
import os

import torch
import torch.nn.functional as F

from enigma_net.ce_approximator.data_gen import generate_dataset
from enigma_net.ce_approximator.model import load_ce_approximator
from enigma_net.ce_approximator.train import downstream_accuracy
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
REC_THRESH = 0.5     # a window is "wrong" if denoiser recovery is below this
GAP_THRESH = 0.15    # a window is a "near-fixed-point" if mean ||d - q|| is below this
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pearson(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    return float((xm * ym).sum() / (xm.norm() * ym.norm() + 1e-8))


def spearman(x, y):
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    return pearson(rx, ry)


def _grad_wrt_logits(loss_scalar_per_window, logits):
    g, = torch.autograd.grad(loss_scalar_per_window.sum(), logits, retain_graph=False)
    return g


def _cos(g, g_true):
    B = g.shape[0]
    return F.cosine_similarity(g.reshape(B, -1), g_true.reshape(B, -1), dim=1)


def per_bin_report(name, values, axis, n_bins):
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

    ngram_loss = NgramLoss(load_ngram_logprobs(NGRAM_PATH, n, device), tau=tau).to(device)
    lm_paths = sorted(glob.glob(os.path.join(LM_DIR, "transformer_lm_*.pth")))
    tf_loss = TransformerLoss(load_transformer_lm(lm_paths[-1], device), tau=tau)

    print("\nGenerating held-out evaluation windows...")
    X, Y, C, P, S, CEW = generate_dataset(
        config, corpus, char_to_idx, device,
        block_size=block_size, windows_per_candidate=4,
        n_random=100, n_traj=40, traj_snapshots=8, n_near=100, n_adv=100,
    )

    rec, ce_pred, ce_true, gaps = [], [], [], []
    cos_ce, cos_tf, cos_ng = [], [], []

    B = 32
    for s in range(0, len(X), B):
        xb = X[s:s + B].to(device)
        yb = Y[s:s + B].to(device)
        cb = C[s:s + B].to(device)
        pb = P[s:s + B].to(device)
        sb = S[s:s + B].to(device)

        with torch.no_grad():
            q = approx.predict_target(xb, cb, pb, sb)
            rec.append((q.argmax(-1) == yb).float().mean(dim=1).cpu())
            ce_pred.append(approx(xb, cipher=cb, positions=pb, qnet_state=sb).cpu())
            ce_true.append(F.cross_entropy(
                xb.reshape(-1, n), yb.reshape(-1), reduction="none"
            ).reshape(xb.shape[0], -1).mean(1).cpu())
            d = torch.softmax(xb / tau, dim=-1)
            gaps.append((d - q).norm(dim=-1).mean(dim=1).cpu())

        lg = xb.detach().requires_grad_(True)
        ce_per = F.cross_entropy(lg.reshape(-1, n), yb.reshape(-1), reduction="none").reshape(lg.shape[0], -1).mean(1)
        g_true = _grad_wrt_logits(ce_per, lg)

        lg = xb.detach().requires_grad_(True)
        g_ce = _grad_wrt_logits(approx(lg, cipher=cb, positions=pb, qnet_state=sb), lg)
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
    gaps    = torch.cat(gaps)
    cos_ce  = torch.cat(cos_ce)
    cos_tf  = torch.cat(cos_tf)
    cos_ng  = torch.cat(cos_ng)

    wrong = rec < REC_THRESH
    near_fixed = gaps < GAP_THRESH
    fm_indicator = (wrong & near_fixed).float()
    fm_rate = float(fm_indicator.sum() / wrong.float().sum().clamp(min=1.0))

    print("\n=== Overall ===")
    print(f"  recovery accuracy : {rec.mean():.4f}")
    print(f"  value  MSE={F.mse_loss(ce_pred, ce_true):.4f}  MAE={(ce_pred-ce_true).abs().mean():.4f}  "
          f"Pearson={pearson(ce_pred, ce_true):.4f}  Spearman={spearman(ce_pred, ce_true):.4f}")
    print(f"  monotonicity Spearman :  loss vs true-CE={spearman(ce_pred, ce_true):+.4f}  "
          f"loss vs recovery={spearman(ce_pred, rec):+.4f} (want strongly negative)")
    print(f"  grad cosine vs true-CE :  CE_approx={cos_ce.mean():+.4f}  "
          f"transformer={cos_tf.mean():+.4f}  ngram={cos_ng.mean():+.4f}")
    print(f"  FALSE-MINIMUM rate (wrong & near-fixed ||d-q||<{GAP_THRESH}): {fm_rate:.4f}"
          f"  | mean gap={gaps.mean():.4f}")

    per_bin_report("Recovery accuracy", rec, CEW, N_CE_BINS)
    per_bin_report("Fixed-point gap ||d-q||", gaps, CEW, N_CE_BINS)
    per_bin_report("False-minimum rate", fm_indicator, CEW, N_CE_BINS)
    per_bin_report("CE_approx grad cosine", cos_ce, CEW, N_CE_BINS)
    per_bin_report("transformer grad cosine", cos_tf, CEW, N_CE_BINS)
    per_bin_report("ngram grad cosine", cos_ng, CEW, N_CE_BINS)

    print("\n=== Downstream short-attack (fresh random keys) ===")
    runs = [
        downstream_accuracy(approx.denoiser, config, corpus, char_to_idx, tau, block_size, n,
                            steps=200, lr=1e-3, seq_len=block_size * 8)
        for _ in range(3)
    ]
    print(f"  mean monitor accuracy over {len(runs)} runs: {sum(runs) / len(runs):.4f}  "
          f"(runs: {[round(a, 3) for a in runs]})")


if __name__ == "__main__":
    main()
