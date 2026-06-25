import os
import random
import itertools
from datetime import datetime
import torch

from enigma_net.fourier.continuous_q_net import ContinuousQNet, permutation_regularizer
from enigma_net.fourier.config import alphabet26_config
from enigma_net import NgramLoss, load_ngram_logprobs
from comparison.continuous_comparison import compare
from visualiser.continuous_visualise import visualise_continuous

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_MODE = "ngram"
TAU = 0.5
Q_LR = 0.01
TOTAL_STEPS = 2500
LEN_STRING = 1000
VAL_LEN = 1000
BATCH_C = 500
BATCH_T = 500
K = 10
WINDOW = 20
STABILITY_EPS = 0.005
PERM_LAMBDA = 0.1
FORCE_TRUE_ACTIVE = True

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NGRAM_PATH = os.path.join(_ROOT, "language", "ngram", "3grams.pth")
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")
MODELS_DIR = os.path.join(_ROOT, "models")

config = alphabet26_config.enigma_config
n = len(config.alphabet)
char_to_idx = {c: i for i, c in enumerate(config.alphabet)}
num_rotors = len(config.rotors)

loss_fn = NgramLoss(load_ngram_logprobs(NGRAM_PATH, n, device), tau=TAU).to(device)

true_positions = [random.randint(0, n - 1) for _ in range(num_rotors)]
print(f"True positions: {true_positions}")

target_enigma = config.build(true_positions)
all_positions = list(itertools.product(range(n), repeat=num_rotors))
C_init = len(all_positions)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = "".join(ch for ch in f.read() if ch in char_to_idx)

def sample_english(length):
    start = random.randint(0, len(corpus) - length - 1)
    return corpus[start:start + length]

def make_data(length):
    plaintext = sample_english(length)
    target_enigma.reset(true_positions)
    ciphertext = target_enigma.encrypt(plaintext)
    in_idx = [char_to_idx[c] for c in ciphertext]
    t_tensor = torch.tensor([char_to_idx[c] for c in plaintext], dtype=torch.long, device=device)
    return in_idx, t_tensor

net = ContinuousQNet(config, initial_positions=all_positions).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=Q_LR)

val_input_indices, val_targets_tensor = make_data(VAL_LEN)

def evaluate_validation(net_model, val_in_idx, val_targets_t, batch_c=BATCH_C):
    C = net_model.num_candidates
    T = len(val_in_idx)
    val_scores = torch.empty(C, device=val_targets_t.device)
    val_ngram = torch.empty(C, device=val_targets_t.device)

    with torch.no_grad():
        for start in range(0, C, batch_c):
            c_indices = torch.arange(start, min(start + batch_c, C), device=val_targets_t.device)
            logits_btn = net_model.encrypt_sequence_slice(val_in_idx, c_indices).transpose(0, 1)
            out_idx = torch.argmax(logits_btn, dim=-1)
            val_scores[c_indices] = (out_idx == val_targets_t.unsqueeze(0)).float().mean(dim=1)
            val_ngram[c_indices] = loss_fn(logits_btn)

    return val_scores, val_ngram

def batched_forward_backward(net, input_indices, active_mask, batch_c, batch_t):
    C = net.num_candidates
    T = len(input_indices)
    loss_per_candidate = torch.zeros(C, device=active_mask.device)

    for c_start in range(0, C, batch_c):
        c_indices = torch.arange(c_start, min(c_start + batch_c, C), device=active_mask.device)
        active_batch = active_mask[c_indices]
        if not active_batch.any():
            continue

        logits_btn = net.encrypt_sequence_slice(input_indices, c_indices).transpose(0, 1)
        loss_b = loss_fn(logits_btn)
        (loss_b * active_batch.float()).sum().backward()
        loss_per_candidate[c_indices] = loss_b.detach()

        del logits_btn, loss_b
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return loss_per_candidate

print("Starting multi-basin parallel optimization...")
K_val = min(K, C_init)
active_mask = torch.ones(C_init, dtype=torch.bool, device=device)
true_candidate_idx = all_positions.index(tuple(true_positions))

step_losses = []
val_losses_history = []

for step in range(TOTAL_STEPS):
    net.train()
    input_indices, _ = make_data(LEN_STRING)

    optimizer.zero_grad()
    loss_per_candidate = batched_forward_backward(net, input_indices, active_mask, BATCH_C, BATCH_T)

    if PERM_LAMBDA > 0:
        perm_loss = PERM_LAMBDA * permutation_regularizer(net)
        (perm_loss * active_mask.float()).sum().backward()
        loss_per_candidate = loss_per_candidate + perm_loss.detach()

    step_losses.append(loss_per_candidate.detach())

    with torch.no_grad():
        for rotor in net.rotors:
            if rotor.Q_real.grad is not None:
                rotor.Q_real.grad[~active_mask] = 0.0
            if rotor.Q_imag.grad is not None:
                rotor.Q_imag.grad[~active_mask] = 0.0
        for p in optimizer.state:
            if isinstance(p, torch.Tensor) and p.ndim > 0 and p.shape[0] == net.num_candidates:
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    state['exp_avg'][~active_mask] = 0.0
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'][~active_mask] = 0.0

    optimizer.step()

    if step % 10 == 0:
        val_scores, val_ngram = evaluate_validation(net, val_input_indices, val_targets_tensor)
        val_losses_history.append(val_ngram.detach())

        if step >= WINDOW:
            loss_decrease = step_losses[-WINDOW] - step_losses[-1]
        elif len(step_losses) > 1:
            loss_decrease = step_losses[0] - step_losses[-1]
        else:
            loss_decrease = torch.zeros(C_init, device=device)

        candidates_metrics = [
            {'index': c, 'val_ngram': val_ngram[c].item(), 'loss_decrease': loss_decrease[c].item()}
            for c in range(C_init)
        ]
        candidates_metrics.sort(key=lambda x: (x['val_ngram'], -x['loss_decrease']))

        topk_indices = [m['index'] for m in candidates_metrics[:K_val]]
        active_mask = torch.zeros(C_init, dtype=torch.bool, device=device)
        active_mask[topk_indices] = True
        if FORCE_TRUE_ACTIVE:
            active_mask[true_candidate_idx] = True

        active_losses = torch.where(active_mask, loss_per_candidate, torch.full_like(loss_per_candidate, float('inf')))
        best_idx = torch.argmin(active_losses).item()
        orig_pos = all_positions[best_idx]
        print(f"Step {step:>4d} | Active: {active_mask.sum().item()} | Best (Orig: {orig_pos}): Loss = {loss_per_candidate[best_idx]:.4f}, MonitorAcc = {val_scores[best_idx]:.3f}")

    if len(val_losses_history) >= 3:
        recent_val = torch.stack(val_losses_history[-3:])
        stds = torch.std(recent_val, dim=0)
        best_idx = torch.argmin(val_losses_history[-1]).item()
        if stds[best_idx].item() < STABILITY_EPS and not FORCE_TRUE_ACTIVE:
            print(f"Step {step:>4d} | Converged. Early stopping.")
            break

print("Training complete.")
val_scores, val_ngram = evaluate_validation(net, val_input_indices, val_targets_tensor)
winner_idx = torch.argmin(val_ngram).item()
winner_orig_pos = all_positions[winner_idx]
print(f"Winner (Orig: {winner_orig_pos}): N-gram Loss = {val_ngram[winner_idx]:.4f}, Acc = {val_scores[winner_idx]:.4f}")

net.prune_candidates([winner_idx])
net.eval()

test_plaintext = sample_english(n ** 2)
target_enigma.reset(true_positions)
test_cipher = target_enigma.encrypt(test_plaintext)
learner_decrypted = net.encrypt_string(test_cipher, greedy=True)
matches = sum(a == b for a, b in zip(test_plaintext, learner_decrypted))
print(f"Accuracy: {matches}/{n**2} ({100 * matches / (n ** 2):.1f}%)")

os.makedirs(MODELS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
weights_path = os.path.join(MODELS_DIR, f"continuous_qnet_{timestamp}.pth")
torch.save(net.state_dict(), weights_path)

compare(weights_path, config=config)
visualise_continuous(net, config.build(), true_positions=true_positions, show_numbers=(n <= 10))
