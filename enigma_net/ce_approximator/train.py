import glob
import os
import random
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from enigma_net.ce_approximator.data_gen import (
    corpus_unigram_prior,
    generate_dataset,
    generate_onpolicy_candidates,
    make_random_qnet,
    make_random_target,
)
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
TAU_AUG          = (0.4, 0.8)
DROPOUT          = 0.1
POS_DROPOUT      = 0.3
WINDOWS_PER_CAND = 8
N_RANDOM         = 300
N_TRAJ           = 150
TRAJ_SNAPSHOTS   = 10
TRAJ_OPT_STEPS   = 400
TRAJ_LR          = 1e-3
N_NEAR           = 300
N_ADV            = 300

USE_CIPHER    = False
USE_POSITIONS = True
USE_STATE     = True
USE_LM_PRIOR  = False

BLOCK_SIZE_OVERRIDE = 256

DAGGER_ROUNDS         = 2
ONPOLICY_CANDIDATES   = 120
ONPOLICY_ATTACK_STEPS = 150
ONPOLICY_ATTACK_LR    = 1e-3
ONPOLICY_SNAPSHOTS    = 6

CALIBRATE     = False
LABEL_SMOOTH  = 0.05
CALIB_C0   = 2.6
CALIB_CMAX = 3.2585
LAMBDA_SMOOTH = 0.0
SMOOTH_EPS    = 0.5

BATCH_SIZE     = 64
EPOCHS         = 25
ROUND_EPOCHS   = 12
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
VAL_SPLIT      = 0.1
N_CE_BINS      = 10
LAMBDA_GRADCOS = 0.0

CE_BAND_DECAY = (1.0, 1.0)

CONF_THRESH = 0.5
HIGH_CE_FRAC = 0.3
LAMBDA_FM   = 0.5

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


def calibrated_target(y, ce, prior, n, c0=CALIB_C0, cmax=CALIB_CMAX):
    oh = F.one_hot(y, n).float()
    alpha = ((ce - c0) / max(cmax - c0, 1e-6)).clamp(0.0, 1.0).view(-1, 1, 1)
    return (1.0 - alpha) * oh + alpha * prior.view(1, 1, -1)


def make_target(y, ce, prior, n):
    if CALIBRATE:
        return calibrated_target(y, ce, prior, n)
    oh = F.one_hot(y, n).float()
    return (1.0 - LABEL_SMOOTH) * oh + LABEL_SMOOTH / n


@torch.no_grad()
def evaluate(denoiser, loader, tau, n, conf_thresh=CONF_THRESH):
    denoiser.eval()
    accs, coss, ces, cws = [], [], [], []
    for xb, yb, cb, pb, sb, ceb in loader:
        xb, yb, cb, pb, sb = xb.to(device), yb.to(device), cb.to(device), pb.to(device), sb.to(device)
        d = torch.softmax(xb / tau, dim=-1)
        q = torch.softmax(denoiser(d, cb, pb, sb), dim=-1)
        accs.append((q.argmax(-1) == yb).float().mean(dim=1).cpu())
        coss.append(grad_cosine(xb, yb, q, tau, n).cpu())
        conf_wrong = (q.max(-1).values > conf_thresh) & (q.argmax(-1) != yb)
        cws.append(conf_wrong.float().mean(dim=1).cpu())
        ces.append(ceb)
    return torch.cat(accs), torch.cat(coss), torch.cat(ces), torch.cat(cws)


def false_min_proxy(confwrong, ce, frac=HIGH_CE_FRAC):
    k = max(1, int(len(ce) * frac))
    hi = ce.argsort(descending=True)[:k]
    return float(confwrong[hi].mean())


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


def make_loaders(trX, trY, trC, trP, trS, trCE, vaX, vaY, vaC, vaP, vaS, vaCE):
    tr_bin, _ = quantile_bins(trCE, N_CE_BINS)
    counts = torch.bincount(tr_bin, minlength=N_CE_BINS).clamp(min=1)
    band = torch.linspace(CE_BAND_DECAY[0], CE_BAND_DECAY[1], N_CE_BINS)
    weights = (band[tr_bin] / counts[tr_bin]).double()
    sampler = WeightedRandomSampler(weights, num_samples=len(trX), replacement=True)
    train_loader = DataLoader(
        TensorDataset(trX, trY, trC, trP, trS, trCE), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(
        TensorDataset(vaX, vaY, vaC, vaP, vaS, vaCE), batch_size=BATCH_SIZE)
    return train_loader, val_loader


def train_round(denoiser, prior, train_loader, val_loader, config, corpus, char_to_idx,
                block_size, n, epochs, tag=""):
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score, best_state = -1e9, None
    for epoch in range(1, epochs + 1):
        denoiser.train()
        epoch_loss, seen = 0.0, 0
        for xb, yb, cb, pb, sb, ceb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            cb, pb, sb, ceb = cb.to(device), pb.to(device), sb.to(device), ceb.to(device)
            tau_b = random.uniform(*TAU_AUG)
            d = torch.softmax(xb / tau_b, dim=-1)
            pred = denoiser(d, cb, pb, sb)
            tgt = make_target(yb, ceb, prior, n)
            loss = -(tgt * F.log_softmax(pred, dim=-1)).sum(-1).mean()
            if LAMBDA_GRADCOS > 0:
                q = torch.softmax(pred, dim=-1)
                loss = loss + LAMBDA_GRADCOS * (1.0 - grad_cosine(xb, yb, q, tau_b, n)).mean()
            if LAMBDA_SMOOTH > 0:
                d2 = torch.softmax((xb + SMOOTH_EPS * torch.randn_like(xb)) / tau_b, dim=-1)
                pred2 = denoiser(d2, cb, pb, sb)
                loss = loss + LAMBDA_SMOOTH * F.mse_loss(
                    torch.softmax(pred2, -1), torch.softmax(pred, -1).detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb); seen += len(xb)
        scheduler.step()

        acc, cos, ce_axis, cw = evaluate(denoiser, val_loader, TAU, n)
        mean_cos, mean_acc, fm = float(cos.mean()), float(acc.mean()), false_min_proxy(cw, ce_axis)

        dwn = None
        if SELECT_BY_DOWNSTREAM and (epoch % DOWNSTREAM_EVERY == 0 or epoch == epochs):
            dwn = downstream_accuracy(
                denoiser, config, corpus, char_to_idx, TAU, block_size, n,
                steps=DOWNSTREAM_STEPS, lr=DOWNSTREAM_LR, seq_len=block_size * DOWNSTREAM_WINDOWS)

        score = (dwn if dwn is not None else mean_cos) - LAMBDA_FM * fm
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in denoiser.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            dwn_str = f"  downstream={dwn:.4f}" if dwn is not None else ""
            print(f"[{tag}] epoch {epoch:>3d}/{epochs}  loss={epoch_loss/seen:.4f}  "
                  f"recovery={mean_acc:.4f}  grad_cos={mean_cos:.4f}  false_min={fm:.4f}"
                  f"  score={score:.4f}{dwn_str}")

    if best_state is not None:
        denoiser.load_state_dict(best_state)
    return best_score


def main():
    config      = alphabet26_config.enigma_config
    n           = len(config.alphabet)
    n_rotors    = len(config.rotors)
    char_to_idx = {c: i for i, c in enumerate(config.alphabet)}

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = "".join(ch for ch in f.read() if ch in char_to_idx)
    prior = corpus_unigram_prior(corpus, char_to_idx, device)

    feats = DenoiserFeatures(
        use_cipher=USE_CIPHER, use_positions=USE_POSITIONS,
        use_state=USE_STATE, use_lm_prior=USE_LM_PRIOR,
        pos_dropout=POS_DROPOUT, num_rotors=n_rotors, n=n,
    )

    lm_paths = sorted(glob.glob(os.path.join(LM_DIR, "transformer_lm_*.pth")))
    if not lm_paths:
        raise FileNotFoundError(f"No transformer LM checkpoint found in {LM_DIR}")
    print(f"Warm-starting denoiser from {lm_paths[-1]}  | features={feats}")
    denoiser = PlaintextDenoiser.from_pretrained_lm(
        lm_paths[-1], feats, device, dropout=DROPOUT, block_size=BLOCK_SIZE_OVERRIDE)
    block_size = denoiser.cfg.block_size
    print(f"Denoiser context (block_size) = {block_size}")

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

    for rnd in range(DAGGER_ROUNDS + 1):
        if rnd > 0:
            print(f"\n=== DAgger round {rnd}: generating on-policy candidates ===")
            approx = CEApproximator(denoiser, tau=TAU, block_size=block_size).to(device)
            oX, oY, oC, oP, oS, oCE = generate_onpolicy_candidates(
                approx, config, corpus, char_to_idx, device,
                block_size=block_size, windows_per_candidate=WINDOWS_PER_CAND,
                n_candidates=ONPOLICY_CANDIDATES, attack_steps=ONPOLICY_ATTACK_STEPS,
                attack_lr=ONPOLICY_ATTACK_LR, snapshots=ONPOLICY_SNAPSHOTS,
            )
            trX = torch.cat([trX, oX]); trY = torch.cat([trY, oY]); trC = torch.cat([trC, oC])
            trP = torch.cat([trP, oP]); trS = torch.cat([trS, oS]); trCE = torch.cat([trCE, oCE])

        epochs = EPOCHS if rnd == 0 else ROUND_EPOCHS
        train_loader, val_loader = make_loaders(
            trX, trY, trC, trP, trS, trCE, vaX, vaY, vaC, vaP, vaS, vaCE)
        print(f"\nTraining round {rnd}/{DAGGER_ROUNDS} for {epochs} epochs on {len(trX)} windows "
              f"(val {len(vaX)}) | block_size={block_size} | device={device}\n")
        train_round(denoiser, prior, train_loader, val_loader, config, corpus, char_to_idx,
                    block_size, n, epochs, tag=f"r{rnd}")

    acc, cos, ce_axis, cw = evaluate(denoiser, val_loader, TAU, n)
    print("\nValidation by true-CE level (high CE = sharp-wrong .. mid = soft blur .. low = solved):")
    print(f"  {'bin':>3} {'CE range':>13} {'count':>6} {'recovery':>9} {'grad_cos':>9} {'conf_wrong':>10}")
    acc_bins = per_bin(acc, ce_axis, N_CE_BINS)
    cos_bins = per_bin(cos, ce_axis, N_CE_BINS)
    cw_bins  = per_bin(cw, ce_axis, N_CE_BINS)
    for (b, lo, hi, c, a), (_, _, _, _, g), (_, _, _, _, w) in zip(acc_bins, cos_bins, cw_bins):
        print(f"  {b:>3} {f'{lo:.2f}-{hi:.2f}':>13} {c:>6} {a:>9.4f} {g:>9.4f} {w:>10.4f}")
    print(f"\nFalse-minimum proxy (confident-wrong rate on most-wrong {int(HIGH_CE_FRAC*100)}% "
          f"of windows): {false_min_proxy(cw, ce_axis):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODELS_DIR, f"ce_approximator_{ts}.pth")
    approx = CEApproximator(denoiser, tau=TAU, block_size=block_size).to(device)
    save_ce_approximator(approx, path)
    print(f"\nSaved checkpoint -> {path}")


if __name__ == "__main__":
    main()
