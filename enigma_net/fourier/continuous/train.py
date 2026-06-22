import sys
import os
import random
import itertools
from datetime import datetime
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from enigma_net.fourier.continuous.net import ContinuousQNet
from enigma_net import CrossEntropyLoss
from enigma_net.train_config import TrainConfig
from config.alphabet3 import alphabet3
from config.alphabet5 import alphabet5
from config.alphabet10 import alphabet10
from config.alphabet15 import alphabet15
from config.alphabet26 import alphabet26
from comparison.continuous_comparison import compare
from visualiser.continuous_visualise import visualise_continuous

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_config = TrainConfig(
    enigma_config=alphabet3,
    loss_fn=CrossEntropyLoss(),
    trainable_rotors=None,
    trainable_reflector=False,
)

PHI_LR = 0.1
Q_LR = 0.01
TOTAL_STEPS = 2500
LOG_STEP = 10
LEN_STRING = 1000
VAL_LEN = 1000
BATCH_C = 500
BATCH_T = 500
K = 10
WINDOW = 20
STABILITY_EPS = 0.005

config = train_config.enigma_config
n = len(config.alphabet)
char_to_idx = {c: i for i, c in enumerate(config.alphabet)}
num_rotors = len(config.rotors)

true_positions = [random.randint(0, n - 1) for _ in range(num_rotors)]
print(f"True positions of the target Enigma: {true_positions}")

target_enigma = config.build(true_positions)
all_positions = list(itertools.product(range(n), repeat=num_rotors))
C_init = len(all_positions)
print(f"Total candidate starting positions (basins): {C_init}")

print(f"Total training steps: {TOTAL_STEPS}")

print(f"Training sequence length: {LEN_STRING}")

net = ContinuousQNet(config, initial_positions=all_positions).to(device)

optimizer = torch.optim.Adam([
    {'params': [net.phi], 'lr': PHI_LR},
    {'params': [p for name, p in net.named_parameters() if name != 'phi'], 'lr': Q_LR}
])

active_original_positions = list(all_positions)
step_losses = []
val_losses_history = []

val_plaintext = "".join(random.choice(config.alphabet) for _ in range(VAL_LEN))
val_input_indices = [char_to_idx[c] for c in val_plaintext]
target_enigma.reset(true_positions)
val_target_indices = [char_to_idx[target_enigma.encrypt_char(c)] for c in val_plaintext]
val_targets_tensor = torch.tensor(val_target_indices, dtype=torch.long, device=device)

def evaluate_validation(net_model, val_in_idx, val_targets_t, batch_c=BATCH_C):
    C = net_model.num_candidates
    T = len(val_in_idx)
    all_logits = []
    with torch.no_grad():
        for start in range(0, C, batch_c):
            c_indices = torch.arange(start, min(start + batch_c, C), device=val_targets_t.device)
            logits_b = net_model.encrypt_sequence_slice(val_in_idx, c_indices)  # [T, B, n]
            all_logits.append(logits_b.transpose(0, 1))                         # [B, T, n]
    val_logits = torch.cat(all_logits, dim=0)  # [C, T, n]

    val_out_idx = torch.argmax(val_logits, dim=-1)
    matches = (val_out_idx == val_targets_t.unsqueeze(0)).sum(dim=1)
    val_scores = matches.float() / T

    flat_logits = val_logits.reshape(C * T, n)
    flat_targets = val_targets_t.repeat(C)
    loss_el = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction='none')
    val_losses = loss_el.reshape(C, T).mean(dim=1)
    return val_scores, val_losses


def batched_forward_backward(net, input_indices, targets, active_mask, batch_c, batch_t):
    """Accumulate gradients across candidate and token mini-batches.

    Loops over (C-batch, T-batch) pairs. Inactive C-batches are skipped.
    Loss is the mean cross-entropy over all T steps (gradient scale matches).
    Does not call zero_grad or optimizer.step. Returns loss_per_candidate [C].
    """
    C = net.num_candidates
    T = len(input_indices)
    loss_per_candidate = torch.zeros(C, device=targets.device)

    # Precompute step offsets once for the full sequence; reused across all batches.
    step_offsets_full = net.precompute_steps(T)  # [T, C, num_rotors]

    for c_start in range(0, C, batch_c):
        c_indices = torch.arange(c_start, min(c_start + batch_c, C), device=targets.device)
        active_batch = active_mask[c_indices]
        if not active_batch.any():
            continue

        step_offsets_c = step_offsets_full[:, c_indices, :]  # [T, B, num_rotors]
        loss_sum_b = torch.zeros(len(c_indices), device=targets.device)

        for t_start in range(0, T, batch_t):
            t_end = min(t_start + batch_t, T)
            T_b = t_end - t_start

            logits_b = net.encrypt_sequence_slice(
                input_indices[t_start:t_end], c_indices,
                step_offsets_c[t_start:t_end]
            ).transpose(0, 1)  # [B, T_b, n]

            B = logits_b.shape[0]
            flat = logits_b.reshape(B * T_b, n)
            loss_el = torch.nn.functional.cross_entropy(
                flat, targets[t_start:t_end].repeat(B), reduction='none'
            )
            # Sum over T_b; divide by total T so gradient magnitude equals the mean.
            loss_t_b = loss_el.reshape(B, T_b).sum(dim=1) / T  # [B]

            loss_sum_b = loss_sum_b + loss_t_b.detach()
            (loss_t_b * active_batch.float()).sum().backward()

            del logits_b, flat, loss_el, loss_t_b
            torch.cuda.empty_cache()

        loss_per_candidate[c_indices] = loss_sum_b  # already normalised by T

    return loss_per_candidate

print("\nStarting Multi-Basin Parallel Optimization...")
K_val = min(K, C_init)
active_mask = torch.ones(C_init, dtype=torch.bool, device=device)

for step in range(TOTAL_STEPS):
    net.train()
    target_enigma.reset(true_positions)
    plaintext = "".join(random.choice(config.alphabet) for _ in range(LEN_STRING))
    input_indices = [char_to_idx[c] for c in plaintext]
    targets = torch.tensor([char_to_idx[target_enigma.encrypt_char(c)] for c in plaintext], dtype=torch.long, device=device)

    optimizer.zero_grad()
    loss_per_candidate = batched_forward_backward(net, input_indices, targets, active_mask, BATCH_C, BATCH_T)
    step_losses.append(loss_per_candidate.detach())

    with torch.no_grad():
        if net.phi.grad is not None:
            net.phi.grad[~active_mask] = 0.0
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

    if step % LOG_STEP == 0:
        val_scores, val_losses = evaluate_validation(net, val_input_indices, val_targets_tensor)
        val_losses_history.append(val_losses.detach())

        if step >= WINDOW:
            loss_decrease = step_losses[-WINDOW] - step_losses[-1]
        elif len(step_losses) > 1:
            loss_decrease = step_losses[0] - step_losses[-1]
        else:
            loss_decrease = torch.zeros(C_init, device=device)

        candidates_metrics = []
        for c in range(C_init):
            metrics = {
                'index': c,
                'val_score': val_scores[c].item(),
                'loss_decrease': loss_decrease[c].item(),
                'val_loss': val_losses[c].item()
            }
            candidates_metrics.append(metrics)

        candidates_metrics.sort(key=lambda x: (x['val_score'], x['loss_decrease'], -x['val_loss']), reverse=True)
        topk_indices = [m['index'] for m in candidates_metrics[:K_val]]
        active_mask = torch.zeros(C_init, dtype=torch.bool, device=device)
        active_mask[topk_indices] = True

    if step % LOG_STEP == 0 or step == TOTAL_STEPS - 1:
        active_losses = torch.where(active_mask, loss_per_candidate,
                                    torch.full_like(loss_per_candidate, float('inf')))
        best_idx = torch.argmin(active_losses).item()
        best_candidate_loss = loss_per_candidate[best_idx].item()
        best_candidate_pos = net.get_positions()[best_idx] if net.num_candidates > 1 else net.get_positions()
        best_candidate_int_pos = net.get_integer_positions()[best_idx] if net.num_candidates > 1 else net.get_integer_positions()
        orig_pos = active_original_positions[best_idx]

        print(f"Step {step:>4d} | Active Candidates: {active_mask.sum().item():>5d}/{net.num_candidates} | Best Candidate (Orig: {orig_pos}): Loss = {best_candidate_loss:.4f}, Rounded Pos = {best_candidate_int_pos}, Continuous = {[f'{p:.2f}' for p in best_candidate_pos]}")

    if len(val_losses_history) >= 3:
        recent_val = torch.stack(val_losses_history[-3:])
        stds = torch.std(recent_val, dim=0)
        best_idx = torch.argmin(val_losses).item()
        if stds[best_idx].item() < STABILITY_EPS and val_losses[best_idx].item() < 0.05:
            print(f"\nStep {step:>4d} | Best candidate stabilized. Early stopping.")
            break

print("\nTraining complete.")
val_scores, val_losses = evaluate_validation(net, val_input_indices, val_targets_tensor)
winner_idx = torch.argmin(val_losses).item()
best_val_score = val_scores[winner_idx].item()
best_val_loss = val_losses[winner_idx].item()
winner_pos = net.get_positions()[winner_idx]
winner_int_pos = net.get_integer_positions()[winner_idx]
winner_orig_pos = active_original_positions[winner_idx]

print(f"Winner Candidate (Orig: {winner_orig_pos}): Loss = {best_val_loss:.4f}, Val Acc = {best_val_score:.4f}, Rounded Pos = {winner_int_pos}, Continuous = {[f'{p:.2f}' for p in winner_pos]}")
print(f"Recovered integer positions: {winner_int_pos}, True: {true_positions}")

print(f"Pruning network down to the winner candidate at index {winner_idx}...")
net.prune_candidates([winner_idx])

print("\nVerification on final test sequence...")
net.eval()
test_plaintext = "".join(random.choice(config.alphabet) for _ in range(n ** 2))
target_enigma.reset(true_positions)
target_encrypted = target_enigma.encrypt(test_plaintext)
learner_encrypted = net.encrypt_string(test_plaintext, greedy=True)
matches = sum(a == b for a, b in zip(target_encrypted, learner_encrypted))
print(f"Accuracy: {matches}/{n**2} ({100 * matches / (n ** 2):.1f}%)")

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
os.makedirs(models_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
weights_path = os.path.join(models_dir, f"continuous_qnet_{timestamp}.pth")
torch.save(net.state_dict(), weights_path)
print(f"Saved weights to: {weights_path}")

print("\nRunning comparison...")
compare(weights_path, config=config)
visualise_continuous(net, config.build(), true_positions=true_positions, show_numbers=(n <= 10))
