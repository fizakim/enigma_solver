"""Train the CE-approximator denoiser.

The denoiser D maps a soft Q-Net output ``d = softmax(logits/tau)`` to per-position logits
over the true plaintext character. It is trained by plain cross-entropy against the known
plaintext, balanced across the permutation-hardness continuum so the near-solution regime is
well resolved. The checkpoint is selected by gradient alignment between the resulting
surrogate loss and true CE — the quantity the Q-Net attack actually consumes.

Run once before using LOSS_MODE = "ce_approximator" in the Q-Net trainer:
    python -m enigma_net.ce_approximator.train
"""

import glob
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from enigma_net.ce_approximator.data_gen import generate_dataset
from enigma_net.ce_approximator.model import (
    CEApproximator,
    PlaintextDenoiser,
    save_ce_approximator,
)
from enigma_net.fourier.config import alphabet26_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CE_DIR     = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")
LM_DIR      = os.path.join(_ROOT, "models")
MODELS_DIR  = os.path.join(_CE_DIR, "models")

TAU            = 0.5          # must match the deploy tau in q_net/train.py
DROPOUT        = 0.1
WINDOWS_PER_CAND = 8
N_RANDOM       = 300
N_TRAJ         = 150
TRAJ_SNAPSHOTS = 10
N_NEAR         = 300
N_ADV          = 300

BATCH_SIZE   = 64
EPOCHS       = 25
LR           = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.1
N_CE_BINS    = 10            # true-CE quantile bins for balanced sampling + reporting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def grad_cosine(logits, y, q, tau, n):
    """Cosine between the true-CE gradient field and the surrogate's, per window.

    true CE grad ∝ softmax(logits) - onehot(y);  surrogate grad ∝ (softmax(logits/tau) - q)/tau.
    Both are computed analytically (no autograd needed). logits/q: [B, L, n], y: [B, L].
    """
    B = logits.shape[0]
    d1 = torch.softmax(logits, dim=-1)
    oh = F.one_hot(y, n).float()
    g_true = (d1 - oh).reshape(B, -1)
    d = torch.softmax(logits / tau, dim=-1)
    g_pred = ((d - q) / tau).reshape(B, -1)
    return F.cosine_similarity(g_true, g_pred, dim=1)   # [B]


@torch.no_grad()
def evaluate(denoiser, loader, tau, n):
    denoiser.eval()
    accs, coss, ces = [], [], []
    for xb, yb, cb in loader:
        xb, yb = xb.to(device), yb.to(device)
        d = torch.softmax(xb / tau, dim=-1)
        q = torch.softmax(denoiser(d), dim=-1)
        accs.append((q.argmax(-1) == yb).float().mean(dim=1).cpu())
        coss.append(grad_cosine(xb, yb, q, tau, n).cpu())
        ces.append(cb)
    return torch.cat(accs), torch.cat(coss), torch.cat(ces)


def quantile_bins(axis, n_bins):
    """Assign each element of `axis` to one of `n_bins` equal-count bins; return (idx, edges)."""
    edges = torch.quantile(axis, torch.linspace(0.0, 1.0, n_bins + 1))
    return torch.bucketize(axis, edges[1:-1].contiguous()), edges


def per_bin(values, axis, n_bins):
    idx, edges = quantile_bins(axis, n_bins)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append((b, float(edges[b]), float(edges[b + 1]), int(m.sum()),
                    float(values[m].mean()) if m.any() else float("nan")))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config      = alphabet26_config.enigma_config
    n           = len(config.alphabet)
    char_to_idx = {c: i for i, c in enumerate(config.alphabet)}

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = "".join(ch for ch in f.read() if ch in char_to_idx)

    lm_paths = sorted(glob.glob(os.path.join(LM_DIR, "transformer_lm_*.pth")))
    if not lm_paths:
        raise FileNotFoundError(f"No transformer LM checkpoint found in {LM_DIR}")
    print(f"Warm-starting denoiser from {lm_paths[-1]}")
    denoiser = PlaintextDenoiser.from_pretrained_lm(lm_paths[-1], device, dropout=DROPOUT)
    block_size = denoiser.cfg.block_size

    X, Y, CE = generate_dataset(
        config, corpus, char_to_idx, device,
        block_size=block_size, windows_per_candidate=WINDOWS_PER_CAND,
        n_random=N_RANDOM, n_traj=N_TRAJ, traj_snapshots=TRAJ_SNAPSHOTS,
        n_near=N_NEAR, n_adv=N_ADV,
    )

    # Split train / val
    N = len(X)
    perm = torch.randperm(N)
    n_val = max(1, int(N * VAL_SPLIT))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    trX, trY, trCE = X[tr_i], Y[tr_i], CE[tr_i]
    vaX, vaY, vaCE = X[val_i], Y[val_i], CE[val_i]

    # True-CE-balanced sampling: equal weight to every stage of the optimization path
    # (sharp-wrong high CE .. soft-blur mid CE .. near-solved low CE), inverse bin frequency
    tr_bin, _ = quantile_bins(trCE, N_CE_BINS)
    counts = torch.bincount(tr_bin, minlength=N_CE_BINS).clamp(min=1)
    weights = (1.0 / counts[tr_bin]).double()
    sampler = WeightedRandomSampler(weights, num_samples=len(trX), replacement=True)

    train_loader = DataLoader(TensorDataset(trX, trY), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(TensorDataset(vaX, vaY, vaCE), batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining denoiser for {EPOCHS} epochs on {len(trX)} windows "
          f"(val {len(vaX)}) | block_size={block_size} | device={device}\n")

    best_cos, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        denoiser.train()
        epoch_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            d = torch.softmax(xb / TAU, dim=-1)
            pred = denoiser(d)
            loss = F.cross_entropy(pred.reshape(-1, n), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb); seen += len(xb)
        scheduler.step()

        acc, cos, ce_axis = evaluate(denoiser, val_loader, TAU, n)
        mean_cos, mean_acc = float(cos.mean()), float(acc.mean())
        if mean_cos > best_cos:
            best_cos = mean_cos
            best_state = {k: v.detach().cpu().clone() for k, v in denoiser.transformer.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3d}/{EPOCHS}  train_ce={epoch_loss/seen:.4f}  "
                  f"val_recovery={mean_acc:.4f}  val_grad_cos={mean_cos:.4f}  best_cos={best_cos:.4f}")

    # Final per-bin report on val with best weights
    denoiser.transformer.load_state_dict(best_state)
    acc, cos, ce_axis = evaluate(denoiser, val_loader, TAU, n)
    print("\nValidation by true-CE level (high CE = sharp-wrong .. mid = soft blur .. low = solved):")
    print(f"  {'bin':>3} {'CE range':>13} {'count':>6} {'recovery':>9} {'grad_cos':>9}")
    acc_bins = per_bin(acc, ce_axis, N_CE_BINS)
    cos_bins = per_bin(cos, ce_axis, N_CE_BINS)
    for (b, lo, hi, c, a), (_, _, _, _, g) in zip(acc_bins, cos_bins):
        print(f"  {b:>3} {f'{lo:.2f}-{hi:.2f}':>13} {c:>6} {a:>9.4f} {g:>9.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODELS_DIR, f"ce_approximator_{ts}.pth")
    approx = CEApproximator(denoiser, tau=TAU, block_size=block_size).to(device)
    save_ce_approximator(approx, path)
    print(f"\nSaved best checkpoint (val_grad_cos={best_cos:.4f}) -> {path}")


if __name__ == "__main__":
    main()
