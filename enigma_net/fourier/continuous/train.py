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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_config = TrainConfig(
    enigma_config=alphabet26,
    loss_fn=CrossEntropyLoss(),
    trainable_rotors=None,
    trainable_reflector=False,
)

PHI_LR = 0.1
Q_LR = 0.01
TOTAL_STEPS = 2500
LOG_STEP = 10
LEN_STRING = len(train_config.enigma_config.alphabet) ** 3
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

VAL_LEN = 1000
val_plaintext = "".join(random.choice(config.alphabet) for _ in range(VAL_LEN))
val_input_indices = [char_to_idx[c] for c in val_plaintext]
target_enigma.reset(true_positions)
val_target_indices = [char_to_idx[target_enigma.encrypt_char(c)] for c in val_plaintext]
val_targets_tensor = torch.tensor(val_target_indices, dtype=torch.long, device=device)

def evaluate_validation(net_model, val_in_idx, val_targets_t):
    with torch.no_grad():
        val_logits = net_model.encrypt_sequence(val_in_idx).transpose(0, 1)
        C_curr = val_logits.shape[0]
        val_out_idx = torch.argmax(val_logits, dim=-1)
        matches = (val_out_idx == val_targets_t.unsqueeze(0)).sum(dim=1)
        val_scores = matches.float() / len(val_in_idx)
        
        flat_logits = val_logits.reshape(C_curr * len(val_in_idx), n)
        flat_targets = val_targets_t.repeat(C_curr)
        loss_el = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction='none')
        val_losses = loss_el.reshape(C_curr, len(val_in_idx)).mean(dim=1)
    return val_scores, val_losses

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
    logits = net.encrypt_sequence(input_indices).transpose(0, 1)
    C_curr = logits.shape[0]
    
    flat_logits = logits.reshape(C_curr * len(plaintext), n)
    flat_targets = targets.repeat(C_curr)
    
    loss_per_element = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction='none')
    loss_per_candidate = loss_per_element.reshape(C_curr, len(plaintext)).mean(dim=1)
    
    step_losses.append(loss_per_candidate.detach())
    
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

    (loss_per_candidate * active_mask.float()).sum().backward()

    with torch.no_grad():
        if net.phi.grad is not None:
            net.phi.grad[~active_mask] = 0.0
        for rotor in net.rotors:
            if rotor.Q_real.grad is not None:
                rotor.Q_real.grad[~active_mask] = 0.0
            if rotor.Q_imag.grad is not None:
                rotor.Q_imag.grad[~active_mask] = 0.0

        for p in optimizer.state:
            if isinstance(p, torch.Tensor) and p.ndim > 0 and p.shape[0] == C_curr:
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    state['exp_avg'][~active_mask] = 0.0
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'][~active_mask] = 0.0

    optimizer.step()

    if step % LOG_STEP == 0 or step == TOTAL_STEPS - 1:
        best_idx = torch.argmin(loss_per_candidate).item()
        best_candidate_loss = loss_per_candidate[best_idx].item()
        best_candidate_pos = net.get_positions()[best_idx] if net.num_candidates > 1 else net.get_positions()
        best_candidate_int_pos = net.get_integer_positions()[best_idx] if net.num_candidates > 1 else net.get_integer_positions()
        orig_pos = active_original_positions[best_idx]
        
        print(f"Step {step:>4d} | Active Candidates: {active_mask.sum().item():>5d}/{C_curr} | Best Candidate (Orig: {orig_pos}): Loss = {best_candidate_loss:.4f}, Rounded Pos = {best_candidate_int_pos}, Continuous = {[f'{p:.2f}' for p in best_candidate_pos]}")

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
