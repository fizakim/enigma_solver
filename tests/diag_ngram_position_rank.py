"""
Diagnostic: are n-grams sufficient to recover the starting position when wirings are known?

Protocol
--------
1. Sample English plaintext from the corpus; encrypt it with the target Enigma at a
   randomly-chosen (or fixed, via SEED) starting position.
2. Build ContinuousQNet over *all* n^R candidate starting positions.
3. Pin every rotor's Q matrix to the ground-truth wiring (no training, no gradient).
4. Evaluate the n-gram loss for every candidate position in batches.
5. Rank candidates and report where the true position falls.

Interpretation
--------------
* Rank 1 of N  -> the signal is sufficient; joint-search (wiring + position) failures
                  are a landscape/optimisation problem, not a missing-signal problem.
* Rank >> 1    -> the n-gram signal is too weak or too flat at this (tau, k, length).
                  Try varying TAU_VALUES / N_GRAM / LEN_STRING below.
"""

import sys
import os
import random
import itertools

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enigma_net.fourier.continuous.net import ContinuousQNet
from enigma_net.fourier.q_net.net import _make_dft
from enigma_net.ngram.loader import load_ngram_logprobs
from enigma_net.ngram.ngram_loss import NgramLoss
from config.alphabet26 import alphabet26
from config.alphabet5 import alphabet5

# ---------------------------------------------------------------------------
# Configuration — edit these to explore
# ---------------------------------------------------------------------------
ALPHABET_CFG = alphabet26   # swap to alphabet5 for a 5-letter smoke test
N_GRAM       = 3            # n-gram order: 1, 2, 3, or 4
TAU_VALUES   = [1.0, 0.5, 0.2, 0.1]   # sweep softmax temperatures
LEN_STRING   = 1000         # ciphertext length
BATCH_C      = 500          # candidates per forward pass (memory budget)
SEED         = 42
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NGRAM_PATH  = os.path.join(_ROOT, "language", "ngram", f"{N_GRAM}grams.pth")
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
random.seed(SEED)

config     = ALPHABET_CFG
n          = len(config.alphabet)
char_to_idx = {c: i for i, c in enumerate(config.alphabet)}
num_rotors = len(config.rotors)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = "".join(ch for ch in f.read() if ch in char_to_idx)
if len(corpus) < 2 * LEN_STRING:
    raise RuntimeError(f"Corpus too small ({len(corpus)} chars); reduce LEN_STRING.")

def sample_english(length):
    start = random.randint(0, len(corpus) - length - 1)
    return corpus[start : start + length]

# True (secret) starting position
true_positions = [random.randint(0, n - 1) for _ in range(num_rotors)]
print(f"Alphabet size : {n}")
print(f"Rotors        : {num_rotors}")
print(f"True positions: {true_positions}")

# Generate ciphertext from real English
target_enigma = config.build(true_positions)
plaintext     = sample_english(LEN_STRING)
target_enigma.reset(true_positions)
cipher_idx    = [char_to_idx[target_enigma.encrypt_char(c)] for c in plaintext]
plain_idx_t   = torch.tensor([char_to_idx[c] for c in plaintext], dtype=torch.long, device=device)
print(f"Plaintext[:40] : {plaintext[:40]}")
print(f"Ciphertext len : {len(cipher_idx)}")

# All candidate starting positions
all_positions = list(itertools.product(range(n), repeat=num_rotors))
C             = len(all_positions)
true_idx      = all_positions.index(tuple(true_positions))
print(f"Total candidates : {C},  true index : {true_idx}")

# ---------------------------------------------------------------------------
# Build the ContinuousQNet and pin Q matrices to ground truth
# ---------------------------------------------------------------------------
net = ContinuousQNet(config, initial_positions=all_positions).to(device)

F_buf, F_inv_buf = _make_dft(n)
F_buf    = F_buf.to(device)
F_inv_buf = F_inv_buf.to(device)

with torch.no_grad():
    for rotor_module, rotor_cfg in zip(net.rotors, config.rotors):
        P = torch.from_numpy(config.wiring_to_matrix(rotor_cfg.wiring)).float().to(device)
        Q_true = F_buf @ P.to(F_buf.dtype) @ F_inv_buf        # [n, n] complex64
        # Broadcast the single true Q to all C candidate slots
        rotor_module.Q_real.copy_(Q_true.real.unsqueeze(0).expand(C, -1, -1))
        rotor_module.Q_imag.copy_(Q_true.imag.unsqueeze(0).expand(C, -1, -1))
        rotor_module.Q_real.requires_grad_(False)
        rotor_module.Q_imag.requires_grad_(False)

print("Q matrices fixed to ground truth (no training).")

# Verify: monitor accuracy at true position should be ~1.0
net.eval()
with torch.no_grad():
    c_true  = torch.tensor([true_idx], device=device)
    logits_t = net.encrypt_sequence_slice(cipher_idx, c_true).transpose(0, 1)  # [1, T, n]
    pred     = logits_t[0].argmax(dim=-1)
    mon_acc  = (pred == plain_idx_t).float().mean().item()
print(f"Monitor accuracy at true position (sanity check, expect ~1.0): {mon_acc:.4f}")
if mon_acc < 0.95:
    print("  WARNING: monitor accuracy is low — check that the Q fix was applied correctly.")

# ---------------------------------------------------------------------------
# Evaluate n-gram loss for every candidate, sweep over TAU values
# ---------------------------------------------------------------------------
log_probs = load_ngram_logprobs(NGRAM_PATH, n, device)

print(f"\n{'='*60}")
print(f"N-gram order: {N_GRAM},  ciphertext length: {LEN_STRING}")
print(f"{'='*60}")

for tau in TAU_VALUES:
    loss_fn = NgramLoss(log_probs, tau=tau).to(device)

    losses = torch.empty(C, device=device)
    with torch.no_grad():
        for c_start in range(0, C, BATCH_C):
            c_end    = min(c_start + BATCH_C, C)
            c_indices = torch.arange(c_start, c_end, device=device)
            logits   = net.encrypt_sequence_slice(cipher_idx, c_indices).transpose(0, 1)  # [B, T, n]
            losses[c_indices] = loss_fn(logits)

    sorted_idx  = torch.argsort(losses)
    true_rank   = (sorted_idx == true_idx).nonzero(as_tuple=True)[0].item()
    loss_true   = losses[true_idx].item()
    loss_best   = losses[sorted_idx[0]].item()
    loss_median = losses[sorted_idx[C // 2]].item()

    # Percentile rank (lower = better, 0% = absolute best)
    percentile = 100.0 * true_rank / C

    print(f"\ntau={tau}")
    print(f"  True position rank : {true_rank + 1:>6d} / {C}  ({percentile:.2f}th percentile)")
    print(f"  Loss at true pos   : {loss_true:.6f}")
    print(f"  Best loss          : {loss_best:.6f}")
    print(f"  Median loss        : {loss_median:.6f}")
    print(f"  Gap (median-true)  : {loss_median - loss_true:.6f}")

    print(f"  Top-10 candidates:")
    for rank, idx in enumerate(sorted_idx[:10].tolist()):
        pos    = all_positions[idx]
        marker = " <-- TRUE" if idx == true_idx else ""
        print(f"    {rank+1:>3d}. pos={list(pos)}  loss={losses[idx].item():.6f}{marker}")

print(f"\n{'='*60}")
print("Done. If the true position consistently ranks #1, the n-gram signal is")
print("sufficient and the problem is purely in the joint wiring+position search.")
print("If it ranks poorly, increase LEN_STRING or switch to a higher N_GRAM order.")
