"""Fully-unsupervised wiring recovery via affine-basin Q-nets.

Positions are NOT cheated.  The initial rotor position is absorbed into the wiring offset:
an affine wiring ``x -> a*x + b (mod n)`` at position 0 is provably identical to the
multiplier wiring ``x -> a*x`` at initial position ``phi`` when ``b = (a-1)*phi (mod n)``.

The search space is all ``(phi(n) * n)^r`` affine wirings (a, b) at position 0.  The target
Enigma also uses an affine wiring at position 0 (b chosen randomly), so the setup is exact.

No gradient descent is used (TOTAL_STEPS=0 default).  The loss function is a pure *ranking*
criterion: evaluate valid affine configs, pick the one with minimum language loss.

**Search strategy** (controlled by ``SEARCH_MODE`` env var):

- ``two_stage`` (default for large n): two-stage search exploiting the factorisation
  ``Q_{a,b} = diag(phase_b) · Q_a``.  Stage 1 scans all ``phi(n)^r`` multiplier combos
  (b=0); Stage 2 exhaustively evaluates ``n^r`` offset combos for each top-K_MULT
  multiplier winner.  For n=26, r=3, K_MULT=50: ~880K evals vs 30M flat — **34× speedup**.
  Runs on CPU.

- ``streaming``: original flat brute-force over all ``(phi(n)·n)^r`` combos (~30M for
  n=26).  GPU recommended.

Run:

    python -m enigma_net.fourier.continuous.affine_basin_train            # two-stage, n=26
    SEARCH_MODE=streaming python -m enigma_net.fourier.continuous.affine_basin_train
    K_MULT=200 python -m enigma_net.fourier.continuous.affine_basin_train  # higher reliability
    ALPHABET=10 python -m enigma_net.fourier.continuous.affine_basin_train # small alphabet
"""

import sys
import os
import random
import itertools
from datetime import datetime

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from enigma_net.fourier.continuous.basins import (
    affine_basin_instances, stream_affine_eval, stream_affine_combo,
    eval_affine_combos, two_stage_affine_eval,
)
from enigma_net.fourier.affine import affine_wiring_string, multiplier_units
from enigma_net import NgramLoss, load_ngram_logprobs
from config.base import EnigmaConfig, RotorConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- config -----------------------------------------------------------------
# alphabet10 by default: (phi(10)*10)^3 = 64,000 candidates — tractable on CPU/GPU.
# Switch to alphabet5 for a quick smoke-test (8,000 candidates).
_ALPHA = os.environ.get("ALPHABET", "26")
if _ALPHA == "5":
    from config.alphabet5 import alphabet5 as config
elif _ALPHA == "10":
    from config.alphabet10 import alphabet10 as config
elif _ALPHA == "15":
    from config.alphabet15 import alphabet15 as config
elif _ALPHA == "26":
    from config.alphabet26 import alphabet26 as config
else:
    raise ValueError(f"Unknown ALPHABET={_ALPHA!r}; choose 5, 10, 15, or 26.")

LOSS_MODE = os.environ.get("LOSS_MODE", "ngram")   # affine basins work well with ngrams
TAU = 0.5

TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 0))   # 0 = pure evaluation (recommended)
LOG_STEP = 10
LEN_STRING = int(os.environ.get("LEN_STRING", 400))
VAL_LEN = int(os.environ.get("VAL_LEN", 400))
BATCH_C = int(os.environ.get("BATCH_C", 512))
K = 16
FORCE_TRUE_ACTIVE = True
# Two-stage search params (used when STREAMING=True and SEARCH_MODE=two_stage).
# K_MULT: multiplier combos carried into Stage 2. 50 gives ~880K evals vs 30M (34×).
# Raise to 200 for extra reliability at ~3.5M evals (8.5× speedup, still CPU-friendly).
K_MULT = int(os.environ.get("K_MULT", 50))
# SEARCH_MODE: "two_stage" (default, CPU-friendly) | "streaming" (30M brute-force, GPU).
SEARCH_MODE = os.environ.get("SEARCH_MODE", "two_stage")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NGRAM_PATH = os.path.join(_ROOT, "language", "ngram", "3grams.pth")
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")
MODELS_DIR = os.path.join(_ROOT, "models")

n = len(config.alphabet)
r = len(config.rotors)
char_to_idx = {c: i for i, c in enumerate(config.alphabet)}

# --- build the AFFINE target (random a, random b — no position anywhere) ----
units = multiplier_units(n)
true_affine = tuple((random.choice(units), random.randint(0, n - 1)) for _ in range(r))
print(f"True affine wirings: {true_affine}  (format: (a, b) per rotor, x->ax+b)")

target_rotors = [
    RotorConfig(wiring=affine_wiring_string(a, b, config.alphabet), notch=rc.notch)
    for (a, b), rc in zip(true_affine, config.rotors)
]
target_config = EnigmaConfig(config.alphabet, target_rotors, config.reflector,
                             config.plugboard_pairs)
target_enigma = target_config.build([0] * r)   # all rotors at position 0 — no cheat needed

# --- loss -------------------------------------------------------------------
NGRAM = LOSS_MODE == "ngram"
if NGRAM:
    loss_fn = NgramLoss(load_ngram_logprobs(NGRAM_PATH, n, device), tau=TAU).to(device)
else:
    from transformer.loss import load_transformer_lm, TransformerLoss
    import glob
    lm_path = sorted(glob.glob(os.path.join(MODELS_DIR, f"transformer_lm_a{n}_*.pth")))[-1]
    print(f"Transformer LM: {lm_path}")
    loss_fn = TransformerLoss(load_transformer_lm(lm_path, device), tau=TAU)

# --- English corpus sampler -------------------------------------------------
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = "".join(ch for ch in f.read() if ch in char_to_idx)
if len(corpus) < 2 * max(LEN_STRING, VAL_LEN):
    raise RuntimeError(f"Corpus too small ({len(corpus)} chars).")


def sample_english(length):
    start = random.randint(0, len(corpus) - length - 1)
    return corpus[start:start + length]


def make_data(length):
    plaintext = sample_english(length)
    target_enigma.reset([0] * r)
    input_indices = [char_to_idx[target_enigma.encrypt_char(c)] for c in plaintext]
    monitor = [char_to_idx[c] for c in plaintext]
    return input_indices, torch.tensor(monitor, dtype=torch.long, device=device)


# --- choose pre-alloc or streaming based on memory estimate -----------------
_P = len(multiplier_units(n)) * n           # phi(n)*n per-rotor pairs
_C = _P ** r                                # total candidates
_mem_gb = _C * r * n * n * 8 / 1024 ** 3   # GB for all Q matrices
STREAMING = _mem_gb > 4.0

print(f"Affine basins (phi(n)*n)^r = {_P}^{r} = {_C:,}  (~{_mem_gb:.1f} GB)")
print(f"Mode: {'STREAMING (lazy Q-table)' if STREAMING else 'PRE-ALLOC'}")

if not STREAMING:
    net = affine_basin_instances(config).to(device)
    C_init = net.num_candidates
    true_basin_idx = net.basin_combos.index(true_affine)
    print(f"True basin index: {true_basin_idx}  combo: {true_affine}")

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    def evaluate_validation(val_in_idx, val_targets_t, batch_c=BATCH_C):
        C = net.num_candidates
        val_scores = torch.empty(C, device=val_targets_t.device)
        val_select = torch.empty(C, device=val_targets_t.device)
        with torch.no_grad():
            for s in range(0, C, batch_c):
                c_idx = torch.arange(s, min(s + batch_c, C), device=val_targets_t.device)
                logits_btn = net.encrypt_sequence_slice(val_in_idx, c_idx).transpose(0, 1)
                val_scores[c_idx] = (
                    torch.argmax(logits_btn, dim=-1) == val_targets_t.unsqueeze(0)
                ).float().mean(dim=1)
                val_select[c_idx] = loss_fn(logits_btn)
        return val_scores, val_select

    def forward_backward(input_indices, active_mask, batch_c=BATCH_C):
        C = net.num_candidates
        loss_per = torch.zeros(C, device=device)
        for s in range(0, C, batch_c):
            c_idx = torch.arange(s, min(s + batch_c, C), device=device)
            active_batch = active_mask[c_idx]
            if not active_batch.any():
                continue
            logits_btn = net.encrypt_sequence_slice(input_indices, c_idx).transpose(0, 1)
            loss_b = loss_fn(logits_btn)
            (loss_b * active_batch.float()).sum().backward()
            loss_per[c_idx] = loss_b.detach()
            del logits_btn, loss_b
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return loss_per

    print("\nEvaluating affine basins (positions fully absorbed into wiring)...")
    K_val = min(K, C_init)
    active_mask = torch.ones(C_init, dtype=torch.bool, device=device)
    val_in_idx, val_targets = make_data(VAL_LEN)

    for step in range(TOTAL_STEPS):
        net.train()
        input_indices, _ = make_data(LEN_STRING)
        optimizer.zero_grad()
        loss_per = forward_backward(input_indices, active_mask)
        with torch.no_grad():
            for rotor in net.rotors:
                if rotor.Q_real.grad is not None:
                    rotor.Q_real.grad[~active_mask] = 0.0
                if rotor.Q_imag.grad is not None:
                    rotor.Q_imag.grad[~active_mask] = 0.0
            for p in optimizer.state:
                if (isinstance(p, torch.Tensor) and p.ndim > 0
                        and p.shape[0] == net.num_candidates):
                    st = optimizer.state[p]
                    if 'exp_avg' in st:
                        st['exp_avg'][~active_mask] = 0.0
                    if 'exp_avg_sq' in st:
                        st['exp_avg_sq'][~active_mask] = 0.0
        optimizer.step()

        if step % LOG_STEP == 0 or step == TOTAL_STEPS - 1:
            val_scores, val_select = evaluate_validation(val_in_idx, val_targets)
            order = sorted(range(C_init), key=lambda c: val_select[c].item())
            topk = order[:K_val]
            active_mask = torch.zeros(C_init, dtype=torch.bool, device=device)
            active_mask[topk] = True
            true_rank = order.index(true_basin_idx)
            survived = "survives" if true_rank < K_val else "PRUNED"
            best_idx = order[0]
            print(f"Step {step:>4d} | true rank {true_rank:>5d}/{C_init} ({survived}), "
                  f"true_select={val_select[true_basin_idx].item():.4f} "
                  f"true_acc={val_scores[true_basin_idx].item():.3f} | "
                  f"best {net.basin_combos[best_idx]} "
                  f"select={val_select[best_idx].item():.4f} "
                  f"acc={val_scores[best_idx].item():.3f}")
            if FORCE_TRUE_ACTIVE:
                active_mask[true_basin_idx] = True

    print("\nEvaluation complete.")
    val_scores, val_select = evaluate_validation(val_in_idx, val_targets)
    winner_idx = int(torch.argmin(val_select).item())
    true_rank = int((val_select < val_select[true_basin_idx]).sum().item())
    winner_combo = net.basin_combos[winner_idx]

    print(f"Winner basin: {winner_combo}")
    print(f"  select={val_select[winner_idx].item():.4f}  acc={val_scores[winner_idx].item():.3f}")
    print(f"True basin:   {true_affine}")
    print(f"  select={val_select[true_basin_idx].item():.4f}  "
          f"acc={val_scores[true_basin_idx].item():.3f}")
    print(f"True basin rank: {true_rank}/{C_init}")

    # Final decrypt check.
    test_plaintext = sample_english(n ** 2)
    target_enigma.reset([0] * r)
    test_cipher = target_enigma.encrypt(test_plaintext)
    decrypted = net.encrypt_string(test_cipher, candidate_idx=winner_idx, greedy=True)
    matches = sum(a == b for a, b in zip(test_plaintext, decrypted))
    print(f"Winner decrypt accuracy: {matches}/{len(test_plaintext)} "
          f"({100 * matches / len(test_plaintext):.1f}%)")

else:
    if TOTAL_STEPS > 0:
        print("WARNING: TOTAL_STEPS>0 is not supported in large-n mode (no GD). "
              "Running pure evaluation.")

    val_in_idx, val_targets = make_data(VAL_LEN)

    if SEARCH_MODE == "two_stage":
        # Two-stage search: ~880K evals for n=26 vs 30M (34× speedup).  CPU-friendly.
        #
        # Stage 1 (phi(n)^r evals): score all multiplier combos (b=0) → keep top-K_MULT.
        #   Even without the true offsets, the correct multiplier tuple outscores wrong
        #   multipliers because the multiplier governs the permutation structure while
        #   offsets only shift characters cyclically.
        # Stage 2 (K_MULT × n^r evals): exhaustively score all n^r offset combos for
        #   each top-K multiplier.  With correct multipliers, language signal is sharp —
        #   this succeeds where coordinate descent fails (coordinate_descent_train.py).
        evals_s1 = len(list(itertools.product(multiplier_units(n), repeat=r)))
        evals_s2 = K_MULT * n ** r
        print(f"\nTwo-stage affine search  (K_mult={K_MULT})")
        print(f"  Stage 1: {evals_s1:,} multiplier combos  |  "
              f"Stage 2: {K_MULT} × {n**r:,} = {evals_s2:,} offset combos")
        print(f"  Total: ~{evals_s1 + evals_s2:,} vs {_C:,} flat — "
              f"{_C / (evals_s1 + evals_s2):.0f}× speedup.  CPU-friendly.")

        winner_combo, winner_loss = two_stage_affine_eval(
            config, val_in_idx, loss_fn,
            K_mult=K_MULT, batch_c=BATCH_C, device=device, verbose=True,
        )

        # Evaluate true combo directly for comparison.
        true_loss = eval_affine_combos(
            config, [true_affine], val_in_idx, loss_fn, batch_c=1, device=device,
        )[0].item()

        print(f"\nWinner basin: {winner_combo}  loss={winner_loss:.4f}")
        print(f"True basin:   {true_affine}  loss={true_loss:.4f}")
        true_rank = None   # not defined in two-stage mode (not all combos evaluated)

    else:
        # SEARCH_MODE == "streaming": original 30M brute-force.
        # For n=26: ~59K batches of 512; runs in minutes on GPU.
        print("\nStreaming evaluation (positions fully absorbed into wiring)...")

        per_rotor, all_losses = stream_affine_eval(
            config, val_in_idx, loss_fn, batch_c=BATCH_C, device=device,
        )
        C_init = len(all_losses)

        P = len(per_rotor)
        ab_to_idx = {ab: i for i, ab in enumerate(per_rotor)}
        true_flat = sum(
            ab_to_idx[true_affine[rot_i]] * (P ** (r - 1 - rot_i))
            for rot_i in range(r)
        )
        true_rank = int((all_losses < all_losses[true_flat]).sum().item())
        winner_flat = int(torch.argmin(all_losses).item())
        winner_combo = stream_affine_combo(winner_flat, per_rotor, r)
        winner_loss = all_losses[winner_flat].item()
        true_loss = all_losses[true_flat].item()

        print(f"Winner basin: {winner_combo}  loss={winner_loss:.4f}")
        print(f"True basin:   {true_affine}  loss={true_loss:.4f}")
        print(f"True basin rank: {true_rank}/{C_init}")

exact_match = winner_combo == true_affine
rank_str = (f", true basin rank {true_rank}" if true_rank is not None else "")
print(f"\nResult: {'MATCH' if exact_match else f'MISMATCH{rank_str}'}")
