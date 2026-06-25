import os
import random
import itertools
import torch

from enigma_net.fourier.basins import (
    affine_basin_instances, stream_affine_eval, stream_affine_combo,
    eval_affine_combos, two_stage_affine_eval,
)
from enigma_net.fourier.affine import affine_wiring_string, multiplier_units
from enigma_net import NgramLoss, load_ngram_logprobs
from config.alphabet26 import alphabet26
from config.base import EnigmaConfig, RotorConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_MODE = "ngram"
TAU = 0.5
TOTAL_STEPS = 0
LEN_STRING = 400
VAL_LEN = 400
BATCH_C = 512
K = 16
K_MULT = 50
SEARCH_MODE = "two_stage"
FORCE_TRUE_ACTIVE = True

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NGRAM_PATH = os.path.join(_ROOT, "language", "ngram", "3grams.pth")
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")
MODELS_DIR = os.path.join(_ROOT, "models")

config = alphabet26
n = len(config.alphabet)
r = len(config.rotors)
char_to_idx = {c: i for i, c in enumerate(config.alphabet)}

units = multiplier_units(n)
true_affine = tuple((random.choice(units), random.randint(0, n - 1)) for _ in range(r))
print(f"True affine wirings: {true_affine}")

target_rotors = [
    RotorConfig(wiring=affine_wiring_string(a, b, config.alphabet), notch=rc.notch)
    for (a, b), rc in zip(true_affine, config.rotors)
]
target_config = EnigmaConfig(config.alphabet, target_rotors, config.reflector, config.plugboard_pairs)
target_enigma = target_config.build([0] * r)

loss_fn = NgramLoss(load_ngram_logprobs(NGRAM_PATH, n, device), tau=TAU).to(device)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = "".join(ch for ch in f.read() if ch in char_to_idx)

def sample_english(length):
    start = random.randint(0, len(corpus) - length - 1)
    return corpus[start:start + length]

def make_data(length):
    plaintext = sample_english(length)
    target_enigma.reset([0] * r)
    input_indices = [char_to_idx[target_enigma.encrypt_char(c)] for c in plaintext]
    monitor = [char_to_idx[c] for c in plaintext]
    return input_indices, torch.tensor(monitor, dtype=torch.long, device=device)

_P = len(multiplier_units(n)) * n
_C = _P ** r
_mem_gb = _C * r * n * n * 8 / 1024 ** 3
STREAMING = _mem_gb > 4.0

if not STREAMING:
    net = affine_basin_instances(config).to(device)
    C_init = net.num_candidates
    true_basin_idx = net.basin_combos.index(true_affine)
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

        if step % 10 == 0 or step == TOTAL_STEPS - 1:
            val_scores, val_select = evaluate_validation(val_in_idx, val_targets)
            order = sorted(range(C_init), key=lambda c: val_select[c].item())
            topk = order[:K]
            active_mask = torch.zeros(C_init, dtype=torch.bool, device=device)
            active_mask[topk] = True
            true_rank = order.index(true_basin_idx)
            best_idx = order[0]
            print(f"Step {step:>4d} | True Rank = {true_rank}/{C_init} | Best: {net.basin_combos[best_idx]} loss={val_select[best_idx].item():.4f}")
            if FORCE_TRUE_ACTIVE:
                active_mask[true_basin_idx] = True

    val_scores, val_select = evaluate_validation(val_in_idx, val_targets)
    winner_idx = int(torch.argmin(val_select).item())
    true_rank = int((val_select < val_select[true_basin_idx]).sum().item())
    winner_combo = net.basin_combos[winner_idx]

    print(f"Winner: {winner_combo}  loss={val_select[winner_idx].item():.4f}")
    print(f"True:   {true_affine}  loss={val_select[true_basin_idx].item():.4f}")

    test_plaintext = sample_english(n ** 2)
    target_enigma.reset([0] * r)
    test_cipher = target_enigma.encrypt(test_plaintext)
    decrypted = net.encrypt_string(test_cipher, candidate_idx=winner_idx, greedy=True)
    matches = sum(a == b for a, b in zip(test_plaintext, decrypted))
    print(f"Accuracy: {matches}/{len(test_plaintext)}")

else:
    val_in_idx, val_targets = make_data(VAL_LEN)

    if SEARCH_MODE == "two_stage":
        winner_combo, winner_loss = two_stage_affine_eval(
            config, val_in_idx, loss_fn,
            K_mult=K_MULT, batch_c=BATCH_C, device=device, verbose=False,
        )
        true_loss = eval_affine_combos(
            config, [true_affine], val_in_idx, loss_fn, batch_c=1, device=device,
        )[0].item()
    else:
        per_rotor, all_losses = stream_affine_eval(
            config, val_in_idx, loss_fn, batch_c=BATCH_C, device=device,
        )
        C_init = len(all_losses)
        P = len(per_rotor)
        ab_to_idx = {ab: i for i, ab in enumerate(per_rotor)}
        true_flat = sum(ab_to_idx[true_affine[rot_i]] * (P ** (r - 1 - rot_i)) for rot_i in range(r))
        true_rank = int((all_losses < all_losses[true_flat]).sum().item())
        winner_flat = int(torch.argmin(all_losses).item())
        winner_combo = stream_affine_combo(winner_flat, per_rotor, r)
        winner_loss = all_losses[winner_flat].item()
        true_loss = all_losses[true_flat].item()
        print(f"True Rank: {true_rank}/{C_init}")

    print(f"Winner: {winner_combo}  loss={winner_loss:.4f}")
    print(f"True:   {true_affine}  loss={true_loss:.4f}")

print(f"Result: {'MATCH' if winner_combo == true_affine else 'MISMATCH'}")
