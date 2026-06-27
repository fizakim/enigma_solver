import glob
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from enigma_net.ce_approximator.data_gen import generate_dataset, make_random_target, make_random_qnet
from enigma_net.ce_approximator.model import (
    CEApproximator,
    DenoiserFeatures,
    PlaintextDenoiser,
    save_ce_approximator,
)
from enigma_net.fourier.config import alphabet26_config

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CE_DIR     = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")
LM_DIR      = os.path.join(_ROOT, "models")
MODELS_DIR  = os.path.join(_CE_DIR, "models")

TAU              = 0.5
DROPOUT          = 0.1
WINDOWS_PER_CAND = 8
N_RANDOM         = 300
N_TRAJ           = 150
TRAJ_SNAPSHOTS   = 10
TRAJ_OPT_STEPS   = 400
TRAJ_LR          = 1e-3
N_NEAR           = 300
N_ADV            = 300

USE_CIPHER    = True
USE_POSITIONS = True
USE_STATE     = True
USE_LM_PRIOR  = False

BATCH_SIZE     = 64
EPOCHS         = 25
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
VAL_SPLIT      = 0.1
N_CE_BINS      = 10
LAMBDA_GRADCOS = 0.0

CE_BAND_DECAY = (1.0, 0.25)

SELECT_BY_DOWNSTREAM = True
DOWNSTREAM_EVERY     = 5
DOWNSTREAM_STEPS     = 150
DOWNSTREAM_LR        = 1e-3
DOWNSTREAM_WINDOWS   = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grad_cosine(logits, y, q, tau, n):
    B = logits.shape[0]
    d1 = torch.softmax(logits, dim=-1)
    oh = F.one_hot(y, n).float()
    g_true = (d1 - oh).reshape(B, -1)
    d = torch.softmax(logits / tau, dim=-1)
    g_pred = ((d - q) / tau).reshape(B, -1)
    return F.cosine_similarity(g_true, g_pred, dim=1)


@torch.no_grad()
def evaluate(denoiser, loader, tau, n):
    denoiser.eval()
    accs, coss, ces = [], [], []
    for xb, yb, cb, pb, sb, ceb in loader:
        xb, yb, cb, pb, sb = xb.to(device), yb.to(device), cb.to(device), pb.to(device), sb.to(device)
        d = torch.softmax(xb / tau, dim=-1)
        q = torch.softmax(denoiser(d, cb, pb, sb), dim=-1)
        accs.append((q.argmax(-1) == yb).float().mean(dim=1).cpu())
        coss.append(grad_cosine(xb, yb, q, tau, n).cpu())
        ces.append(ceb)
    return torch.cat(accs), torch.cat(coss), torch.cat(ces)


def downstream_accuracy(denoiser, config, corpus, char_to_idx, tau, block_size, n,
                        steps, lr, seq_len):
    was_training = denoiser.training
    denoiser.eval()
    n_rotors = len(config.rotors)
    approx = CEApproximator(denoiser, tau=tau, block_size=block_size)

    with torch.no_grad():
        positions = [int(torch.randint(0, n, (1,))) for _ in range(n_rotors)]
        target = make_random_target(config, device)
        start = int(torch.randint(0, len(corpus) - seq_len - 1, (1,)))
        plaintext = corpus[start:start + seq_len]
        plain_list = [char_to_idx[c] for c in plaintext]
        target.reset(positions)
        cipher_idx = target.encrypt_sequence(plain_list).argmax(-1).tolist()

    learner = make_random_qnet(config, device)
    monitor = torch.tensor(plain_list, dtype=torch.long, device=device)
    cipher_t = torch.tensor(cipher_idx, dtype=torch.long, device=device).unsqueeze(0)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)

    accs = []
    for step in range(steps):
        learner.reset(positions)
        logits = learner.encrypt_sequence(cipher_idx)
        pos = learner.step_positions(len(cipher_idx)).unsqueeze(0)
        state = learner.state_features().unsqueeze(0)
        loss = approx(logits.unsqueeze(0), cipher=cipher_t, positions=pos, qnet_state=state).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step >= steps - 10:
            with torch.no_grad():
                accs.append((logits.argmax(-1) == monitor).float().mean().item())

    if was_training:
        denoiser.train()
    return float(sum(accs) / max(1, len(accs)))


def quantile_bins(axis, n_bins):
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


def main():
    config      = alphabet26_config.enigma_config
    n           = len(config.alphabet)
    n_rotors    = len(config.rotors)
    char_to_idx = {c: i for i, c in enumerate(config.alphabet)}

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = "".join(ch for ch in f.read() if ch in char_to_idx)

    feats = DenoiserFeatures(
        use_cipher=USE_CIPHER, use_positions=USE_POSITIONS,
        use_state=USE_STATE, use_lm_prior=USE_LM_PRIOR,
        num_rotors=n_rotors, n=n,
    )

    lm_paths = sorted(glob.glob(os.path.join(LM_DIR, "transformer_lm_*.pth")))
    if not lm_paths:
        raise FileNotFoundError(f"No transformer LM checkpoint found in {LM_DIR}")
    print(f"Warm-starting denoiser from {lm_paths[-1]}  | features={feats}")
    denoiser = PlaintextDenoiser.from_pretrained_lm(lm_paths[-1], feats, device, dropout=DROPOUT)
    block_size = denoiser.cfg.block_size

    X, Y, C, P, S, CE = generate_dataset(
        config, corpus, char_to_idx, device,
        block_size=block_size, windows_per_candidate=WINDOWS_PER_CAND,
        n_random=N_RANDOM, n_traj=N_TRAJ, traj_snapshots=TRAJ_SNAPSHOTS,
        traj_opt_steps=TRAJ_OPT_STEPS, traj_lr=TRAJ_LR, n_near=N_NEAR, n_adv=N_ADV,
    )

    N = len(X)
    perm = torch.randperm(N)
    n_val = max(1, int(N * VAL_SPLIT))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    trX, trY, trC, trP, trS, trCE = X[tr_i], Y[tr_i], C[tr_i], P[tr_i], S[tr_i], CE[tr_i]
    vaX, vaY, vaC, vaP, vaS, vaCE = X[val_i], Y[val_i], C[val_i], P[val_i], S[val_i], CE[val_i]

    tr_bin, _ = quantile_bins(trCE, N_CE_BINS)
    counts = torch.bincount(tr_bin, minlength=N_CE_BINS).clamp(min=1)
    band = torch.linspace(CE_BAND_DECAY[0], CE_BAND_DECAY[1], N_CE_BINS)
    weights = (band[tr_bin] / counts[tr_bin]).double()
    sampler = WeightedRandomSampler(weights, num_samples=len(trX), replacement=True)

    train_loader = DataLoader(TensorDataset(trX, trY, trC, trP, trS), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(TensorDataset(vaX, vaY, vaC, vaP, vaS, vaCE), batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining denoiser for {EPOCHS} epochs on {len(trX)} windows "
          f"(val {len(vaX)}) | block_size={block_size} | device={device}\n")

    best_cos, best_cos_state = -1.0, None
    best_dwn, best_dwn_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        denoiser.train()
        epoch_loss, seen = 0.0, 0
        for xb, yb, cb, pb, sb in train_loader:
            xb, yb, cb, pb, sb = xb.to(device), yb.to(device), cb.to(device), pb.to(device), sb.to(device)
            d = torch.softmax(xb / TAU, dim=-1)
            pred = denoiser(d, cb, pb, sb)
            loss = F.cross_entropy(pred.reshape(-1, n), yb.reshape(-1))
            if LAMBDA_GRADCOS > 0:
                q = torch.softmax(pred, dim=-1)
                loss = loss + LAMBDA_GRADCOS * (1.0 - grad_cosine(xb, yb, q, TAU, n)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb); seen += len(xb)
        scheduler.step()

        acc, cos, ce_axis = evaluate(denoiser, val_loader, TAU, n)
        mean_cos, mean_acc = float(cos.mean()), float(acc.mean())
        if mean_cos > best_cos:
            best_cos = mean_cos
            best_cos_state = {k: v.detach().cpu().clone() for k, v in denoiser.state_dict().items()}

        dwn = None
        if SELECT_BY_DOWNSTREAM and (epoch % DOWNSTREAM_EVERY == 0 or epoch == EPOCHS):
            dwn = downstream_accuracy(
                denoiser, config, corpus, char_to_idx, TAU, block_size, n,
                steps=DOWNSTREAM_STEPS, lr=DOWNSTREAM_LR, seq_len=block_size * DOWNSTREAM_WINDOWS)
            if dwn > best_dwn:
                best_dwn = dwn
                best_dwn_state = {k: v.detach().cpu().clone() for k, v in denoiser.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            dwn_str = f"  downstream_acc={dwn:.4f}" if dwn is not None else ""
            print(f"Epoch {epoch:>3d}/{EPOCHS}  train_loss={epoch_loss/seen:.4f}  "
                  f"val_recovery={mean_acc:.4f}  val_grad_cos={mean_cos:.4f}  best_cos={best_cos:.4f}{dwn_str}")

    if SELECT_BY_DOWNSTREAM and best_dwn_state is not None:
        chosen_state, sel = best_dwn_state, f"downstream_acc={best_dwn:.4f}"
    else:
        chosen_state, sel = best_cos_state, f"val_grad_cos={best_cos:.4f}"
    denoiser.load_state_dict(chosen_state)

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
    print(f"\nSaved chosen checkpoint ({sel}) -> {path}")


if __name__ == "__main__":
    main()
