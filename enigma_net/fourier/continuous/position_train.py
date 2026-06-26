import os
import random
import itertools
import torch

from enigma_net.fourier.position_net import ContinuousPositionNet
from enigma_net import NgramLoss, load_ngram_logprobs
from config.alphabet26 import alphabet26

VAL_LEN = 500
BATCH_C = 512
TAU     = 0.5

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
    return cipher_idx


all_positions = list(itertools.product(range(n), repeat=r))
true_idx      = all_positions.index(tuple(true_positions))

net       = ContinuousPositionNet(config, torch.tensor(all_positions, dtype=torch.float32)).to(device)
val_in    = make_data(VAL_LEN)
losses    = torch.empty(len(all_positions))

with torch.no_grad():
    for s in range(0, len(all_positions), BATCH_C):
        c_idx        = torch.arange(s, min(s + BATCH_C, len(all_positions)))
        logits       = net.encrypt_sequence_slice(val_in, c_idx).transpose(0, 1)
        losses[c_idx] = loss_fn(logits)

order      = losses.argsort().tolist()
winner_idx = order[0]
winner_pos = list(all_positions[winner_idx])
true_rank  = order.index(true_idx) + 1

print(f"True rank: {true_rank}/{len(all_positions)}")
print(f"Winner: {winner_pos}  loss={losses[winner_idx]:.4f}")
print(f"True:   {true_positions}")
print(f"Result: {'MATCH' if winner_pos == true_positions else 'MISMATCH'}")

test_plain  = sample_english(n * n)
target_enigma.reset(true_positions)
test_cipher = target_enigma.encrypt(test_plain)
net.prune_candidates([winner_idx])
decrypted   = net.encrypt_string(test_cipher, candidate_idx=0, greedy=True)
matches     = sum(a == b for a, b in zip(test_plain, decrypted))
print(f"Decryption accuracy: {matches}/{len(test_plain)} ({100 * matches / len(test_plain):.1f}%)")
