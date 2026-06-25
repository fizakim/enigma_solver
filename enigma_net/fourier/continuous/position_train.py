import os
import random
import itertools
import torch

from enigma_net.fourier.position_net import ContinuousPositionNet
from enigma_net import NgramLoss, load_ngram_logprobs
from config.alphabet26 import alphabet26

PHI_LR       = 0.3
PHASE2_STEPS = 30
LEN_STRING   = 500
VAL_LEN      = 500
BATCH_C      = 512
K_PRUNE      = 50
K_FINAL      = 3
TAU          = 0.5

device = torch.device("cpu")

_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NGRAM_PATH = os.path.join(_ROOT, "language", "ngram", "3grams.pth")
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")

config      = alphabet26
n           = len(config.alphabet)
r           = len(config.rotors)
char_to_idx = {c: i for i, c in enumerate(config.alphabet)}

loss_fn = NgramLoss(load_ngram_logprobs(NGRAM_PATH, n, device), tau=TAU).to(device)

with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = "".join(ch for ch in f.read() if ch in char_to_idx)


def sample_english(length):
    s = random.randint(0, len(corpus) - length - 1)
    return corpus[s : s + length]


true_positions = [random.randint(0, n - 1) for _ in range(r)]
print(f"True positions: {true_positions}")
target_enigma = config.build(true_positions)


def make_data(length):
    plaintext = sample_english(length)
    target_enigma.reset(true_positions)
    ciphertext = target_enigma.encrypt(plaintext)
    cipher_idx = [char_to_idx[c] for c in ciphertext]
    plain_idx  = torch.tensor([char_to_idx[c] for c in plaintext], dtype=torch.long)
    return cipher_idx, plain_idx


all_positions = list(itertools.product(range(n), repeat=r))
true_idx      = all_positions.index(tuple(true_positions))

initial_phi = torch.tensor(all_positions, dtype=torch.float32)
net_grid    = ContinuousPositionNet(config, initial_phi).to(device)
val_in, _   = make_data(VAL_LEN)
grid_losses = torch.empty(len(all_positions))

with torch.no_grad():
    for s in range(0, len(all_positions), BATCH_C):
        c_idx  = torch.arange(s, min(s + BATCH_C, len(all_positions)))
        logits = net_grid.encrypt_sequence_slice(val_in, c_idx).transpose(0, 1)
        grid_losses[c_idx] = loss_fn(logits)

order_p1     = grid_losses.argsort().tolist()
top_k_global = order_p1[:K_PRUNE]
print(f"True rank after Phase 1: {order_p1.index(true_idx) + 1}/{len(all_positions)}")

keep_phi = torch.tensor([all_positions[i] for i in top_k_global], dtype=torch.float32)
net_gd   = ContinuousPositionNet(config, keep_phi).to(device)

true_pruned = top_k_global.index(true_idx) if true_idx in top_k_global else -1
optimizer   = torch.optim.Adam([net_gd.phi], lr=PHI_LR)
active_mask = torch.ones(K_PRUNE, dtype=torch.bool)
val_in2, _  = make_data(VAL_LEN)

for step in range(PHASE2_STEPS):
    input_indices, _ = make_data(LEN_STRING)
    optimizer.zero_grad()
    loss_per = torch.zeros(K_PRUNE)

    for s in range(0, K_PRUNE, BATCH_C):
        c_idx  = torch.arange(s, min(s + BATCH_C, K_PRUNE))
        logits = net_gd.encrypt_sequence_slice(input_indices, c_idx).transpose(0, 1)
        loss_b = loss_fn(logits)
        (loss_b * active_mask[c_idx].float()).sum().backward()
        loss_per[c_idx] = loss_b.detach()

    with torch.no_grad():
        if net_gd.phi.grad is not None:
            net_gd.phi.grad[~active_mask] = 0.0
    optimizer.step()

    if step % 5 == 0 or step == PHASE2_STEPS - 1:
        order     = loss_per.argsort().tolist()
        true_rank = order.index(true_pruned) + 1 if true_pruned >= 0 else -1
        best_pos  = (net_gd.phi[order[0]].detach().round().long() % n).tolist()
        print(f"Step {step:>3d} | True rank: {true_rank}/{active_mask.sum()} | Best: {best_pos}")

        if step > 3 * PHASE2_STEPS // 4:
            active_mask = torch.zeros(K_PRUNE, dtype=torch.bool)
            active_mask[order[:K_FINAL]] = True

final_losses = torch.empty(K_PRUNE)
with torch.no_grad():
    for s in range(0, K_PRUNE, BATCH_C):
        c_idx  = torch.arange(s, min(s + BATCH_C, K_PRUNE))
        logits = net_gd.encrypt_sequence_slice(val_in2, c_idx).transpose(0, 1)
        final_losses[c_idx] = loss_fn(logits)

winner_pruned = int(final_losses.argmin())
winner_pos    = list(all_positions[top_k_global[winner_pruned]])

print(f"\nWinner: {winner_pos}  loss={final_losses[winner_pruned]:.4f}")
print(f"True:   {true_positions}")
print(f"Result: {'MATCH' if winner_pos == true_positions else 'MISMATCH'}")

test_plain  = sample_english(n * n)
target_enigma.reset(true_positions)
test_cipher = target_enigma.encrypt(test_plain)
net_gd.prune_candidates([winner_pruned])
decrypted   = net_gd.encrypt_string(test_cipher, candidate_idx=0, greedy=True)
matches     = sum(a == b for a, b in zip(test_plain, decrypted))
print(f"Decryption accuracy: {matches}/{len(test_plain)} ({100 * matches / len(test_plain):.1f}%)")
