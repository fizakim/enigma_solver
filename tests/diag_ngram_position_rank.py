import sys
import os
import random
import itertools
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enigma_net.fourier.continuous_q_net import ContinuousQNet
from enigma_net.fourier.q_net import _make_dft
from enigma_net.ngram.loader import load_ngram_logprobs
from enigma_net.ngram.ngram_loss import NgramLoss
from config.alphabet26 import alphabet26

ALPHABET_CFG = alphabet26
N_GRAM = 3
TAU_VALUES = [1.0, 0.5, 0.2, 0.1]
LEN_STRING = 1000
BATCH_C = 500
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NGRAM_PATH = os.path.join(_ROOT, "language", "ngram", f"{N_GRAM}grams.pth")
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")

torch.manual_seed(SEED)
random.seed(SEED)

config = ALPHABET_CFG
n = len(config.alphabet)
char_to_idx = {c: i for i, c in enumerate(config.alphabet)}
num_rotors = len(config.rotors)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = "".join(ch for ch in f.read() if ch in char_to_idx)
if len(corpus) < 2 * LEN_STRING:
    raise RuntimeError("Corpus too small.")

def sample_english(length):
    start = random.randint(0, len(corpus) - length - 1)
    return corpus[start : start + length]

true_positions = [random.randint(0, n - 1) for _ in range(num_rotors)]
print(f"True positions: {true_positions}")

target_enigma = config.build(true_positions)
plaintext = sample_english(LEN_STRING)
target_enigma.reset(true_positions)
cipher_idx = [char_to_idx[target_enigma.encrypt_char(c)] for c in plaintext]
plain_idx_t = torch.tensor([char_to_idx[c] for c in plaintext], dtype=torch.long, device=device)

all_positions = list(itertools.product(range(n), repeat=num_rotors))
C = len(all_positions)
true_idx = all_positions.index(tuple(true_positions))

net = ContinuousQNet(config, initial_positions=all_positions).to(device)

F_buf, F_inv_buf = _make_dft(n)
F_buf = F_buf.to(device)
F_inv_buf = F_inv_buf.to(device)

with torch.no_grad():
    for rotor_module, rotor_cfg in zip(net.rotors, config.rotors):
        P = torch.from_numpy(config.wiring_to_matrix(rotor_cfg.wiring)).float().to(device)
        Q_true = F_buf @ P.to(F_buf.dtype) @ F_inv_buf
        rotor_module.Q_real.copy_(Q_true.real.unsqueeze(0).expand(C, -1, -1))
        rotor_module.Q_imag.copy_(Q_true.imag.unsqueeze(0).expand(C, -1, -1))
        rotor_module.Q_real.requires_grad_(False)
        rotor_module.Q_imag.requires_grad_(False)

net.eval()
with torch.no_grad():
    c_true = torch.tensor([true_idx], device=device)
    logits_t = net.encrypt_sequence_slice(cipher_idx, c_true).transpose(0, 1)
    pred = logits_t[0].argmax(dim=-1)
    mon_acc = (pred == plain_idx_t).float().mean().item()
print(f"Sanity check monitor acc: {mon_acc:.4f}")

log_probs = load_ngram_logprobs(NGRAM_PATH, n, device)

for tau in TAU_VALUES:
    loss_fn = NgramLoss(log_probs, tau=tau).to(device)
    losses = torch.empty(C, device=device)
    with torch.no_grad():
        for c_start in range(0, C, BATCH_C):
            c_end = min(c_start + BATCH_C, C)
            c_indices = torch.arange(c_start, c_end, device=device)
            logits = net.encrypt_sequence_slice(cipher_idx, c_indices).transpose(0, 1)
            losses[c_indices] = loss_fn(logits)

    sorted_idx = torch.argsort(losses)
    true_rank = (sorted_idx == true_idx).nonzero(as_tuple=True)[0].item()
    print(f"tau={tau} | True Rank = {true_rank+1}/{C} | Loss = {losses[true_idx].item():.6f}")
