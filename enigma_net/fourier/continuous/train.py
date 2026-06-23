import sys
import os
import random
import itertools
from datetime import datetime
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from enigma_net.fourier.continuous.net import ContinuousQNet
from enigma_net.fourier.continuous.perm_reg import permutation_regularizer
from enigma_net import CrossEntropyLoss, NgramLoss, load_ngram_logprobs
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

# ---------------------------------------------------------------------------
# Loss mode: "ce" (supervised cross-entropy, known plaintext) or
#            "ngram" (unsupervised; only ciphertext + an English trigram prior).
#
# In "ngram" mode the data flow is reversed relative to "ce": the plaintext is
# *real English* sampled from the corpus, the target Enigma encrypts it, and the
# NET IS FED THE CIPHERTEXT. The net's output is scored by how English-like it is
# (trigram log-prob). The known plaintext is used only as a monitor-only accuracy
# readout, never for candidate selection or early-stopping.
# ---------------------------------------------------------------------------
LOSS_MODE = "ngram"   # "ce" or "ngram"
NGRAM = LOSS_MODE == "ngram"
TAU = 0.5             # softmax temperature for the n-gram soft decoding (fixed)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NGRAM_PATH = os.path.join(_ROOT, "language", "ngram", "3grams.pth")
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")

train_config = TrainConfig(
    enigma_config=alphabet26,
    loss_fn=CrossEntropyLoss(),
    trainable_rotors=None,
    trainable_reflector=False,
)

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
# Permutation regularizer weight. Penalises Q matrices whose spatial form
# deviates from a valid permutation (doubly-stochastic + binary entries).
# 0 disables it. Start around 0.1; raise if cheating persists.
PERM_LAMBDA = 0.1

config = train_config.enigma_config
n = len(config.alphabet)
char_to_idx = {c: i for i, c in enumerate(config.alphabet)}
num_rotors = len(config.rotors)

# Build the loss function. In ngram mode it owns the trigram table (on device).
if NGRAM:
    loss_fn = NgramLoss(load_ngram_logprobs(NGRAM_PATH, n, device), tau=TAU).to(device)
else:
    loss_fn = train_config.loss_fn

true_positions = [random.randint(0, n - 1) for _ in range(num_rotors)]
print(f"True positions of the target Enigma: {true_positions}")

target_enigma = config.build(true_positions)
all_positions = list(itertools.product(range(n), repeat=num_rotors))
C_init = len(all_positions)
print(f"Total candidate starting positions (basins): {C_init}")
print(f"Loss mode: {LOSS_MODE}")
print(f"Total training steps: {TOTAL_STEPS}")
print(f"Training sequence length: {LEN_STRING}")

# English corpus sampler (ngram mode only): the plaintext must be real English so
# that a correct decryption scores high under the trigram prior.
if NGRAM:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = "".join(ch for ch in f.read() if ch in char_to_idx)
    if len(corpus) < 2 * LEN_STRING:
        raise RuntimeError(f"Corpus too small ({len(corpus)} chars) for LEN_STRING={LEN_STRING}.")

    def sample_english(length):
        start = random.randint(0, len(corpus) - length - 1)
        return corpus[start:start + length]


def make_data(length):
    """Return (input_indices, monitor_target_tensor) for one sequence.

    ce:    net is fed plaintext; monitor target is the ciphertext (the net learns
           to mimic the Enigma).
    ngram: net is fed ciphertext of real English; monitor target is the English
           plaintext (the net learns to decrypt). The loss ignores the target.
    """
    if NGRAM:
        plaintext = sample_english(length)
        target_enigma.reset(true_positions)
        input_indices = [char_to_idx[target_enigma.encrypt_char(c)] for c in plaintext]
        monitor = [char_to_idx[c] for c in plaintext]
    else:
        plaintext = "".join(random.choice(config.alphabet) for _ in range(length))
        input_indices = [char_to_idx[c] for c in plaintext]
        target_enigma.reset(true_positions)
        monitor = [char_to_idx[target_enigma.encrypt_char(c)] for c in plaintext]
    return input_indices, torch.tensor(monitor, dtype=torch.long, device=device)


net = ContinuousQNet(config, initial_positions=all_positions).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=Q_LR)

active_original_positions = list(all_positions)
step_losses = []
val_losses_history = []

# Fixed validation sequence. val_input_indices is what the net is fed;
# val_targets_tensor is the monitor target (plaintext in ngram mode, ciphertext in ce).
val_input_indices, val_targets_tensor = make_data(VAL_LEN)


def evaluate_validation(net_model, val_in_idx, val_targets_t, loss_fn, batch_c=BATCH_C):
    """Returns (val_scores, val_ce, val_ngram).

    val_scores: per-candidate output-vs-monitor accuracy (always; monitor-only).
    val_ce:     per-candidate supervised cross-entropy against the monitor target.
    val_ngram:  per-candidate unsupervised n-gram loss, or None in ce mode.
    """
    C = net_model.num_candidates
    T = len(val_in_idx)
    is_ngram = getattr(loss_fn, "requires_full_sequence", False)
    val_scores = torch.empty(C, device=val_targets_t.device)
    val_ce = torch.empty(C, device=val_targets_t.device)
    val_ngram = torch.empty(C, device=val_targets_t.device) if is_ngram else None

    with torch.no_grad():
        for start in range(0, C, batch_c):
            c_indices = torch.arange(start, min(start + batch_c, C), device=val_targets_t.device)
            logits_btn = net_model.encrypt_sequence_slice(val_in_idx, c_indices).transpose(0, 1)  # [B, T, n]
            B = logits_btn.shape[0]

            out_idx = torch.argmax(logits_btn, dim=-1)                              # [B, T]
            val_scores[c_indices] = (out_idx == val_targets_t.unsqueeze(0)).float().mean(dim=1)

            ce = torch.nn.functional.cross_entropy(
                logits_btn.reshape(B * T, n), val_targets_t.repeat(B), reduction='none'
            ).reshape(B, T).mean(dim=1)
            val_ce[c_indices] = ce

            if is_ngram:
                val_ngram[c_indices] = loss_fn(logits_btn)

    return val_scores, val_ce, val_ngram


def batched_forward_backward(net, input_indices, targets, active_mask, loss_fn, batch_c, batch_t):
    """Accumulate gradients across candidate (and, for ce, token) mini-batches.

    Does not call zero_grad or optimizer.step. Returns loss_per_candidate [C].
    """
    C = net.num_candidates
    T = len(input_indices)
    loss_per_candidate = torch.zeros(C, device=targets.device)

    # ----- Sequential loss (n-gram): full per-candidate sequence, C-batch only -----
    if getattr(loss_fn, "requires_full_sequence", False):
        for c_start in range(0, C, batch_c):
            c_indices = torch.arange(c_start, min(c_start + batch_c, C), device=targets.device)
            active_batch = active_mask[c_indices]
            if not active_batch.any():
                continue

            logits_btn = net.encrypt_sequence_slice(input_indices, c_indices).transpose(0, 1)  # [B, T, n]
            loss_b = loss_fn(logits_btn)  # [B]
            (loss_b * active_batch.float()).sum().backward()
            loss_per_candidate[c_indices] = loss_b.detach()

            del logits_btn, loss_b
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return loss_per_candidate

    # ----- Per-token loss (cross-entropy): T-batched, unchanged -----
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
            if device.type == "cuda":
                torch.cuda.empty_cache()

        loss_per_candidate[c_indices] = loss_sum_b  # already normalised by T

    return loss_per_candidate


print("\nStarting Multi-Basin Parallel Optimization...")
K_val = min(K, C_init)
active_mask = torch.ones(C_init, dtype=torch.bool, device=device)

# --- Falsification test -------------------------------------------------------
# Pin the true-position candidate permanently active and report its rank each
# evaluation. If ngram mode now converges, the failure was the pruning schedule
# (true candidate discarded before its Q trained), not the removal of phi.
FORCE_TRUE_ACTIVE = True
true_candidate_idx = all_positions.index(tuple(true_positions))
print(f"True candidate index: {true_candidate_idx} (positions {tuple(true_positions)})")

for step in range(TOTAL_STEPS):
    net.train()
    input_indices, targets = make_data(LEN_STRING)

    optimizer.zero_grad()
    loss_per_candidate = batched_forward_backward(net, input_indices, targets, active_mask, loss_fn, BATCH_C, BATCH_T)

    if PERM_LAMBDA > 0:
        perm_loss = PERM_LAMBDA * permutation_regularizer(net)        # [C]
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

    if step % LOG_STEP == 0:
        val_scores, val_ce, val_ngram = evaluate_validation(net, val_input_indices, val_targets_tensor, loss_fn)
        # Selection metric (lower is better): unsupervised n-gram loss in ngram mode,
        # supervised cross-entropy otherwise.
        val_select = val_ngram if NGRAM else val_ce
        val_losses_history.append(val_select.detach())

        if step >= WINDOW:
            loss_decrease = step_losses[-WINDOW] - step_losses[-1]
        elif len(step_losses) > 1:
            loss_decrease = step_losses[0] - step_losses[-1]
        else:
            loss_decrease = torch.zeros(C_init, device=device)

        candidates_metrics = []
        for c in range(C_init):
            candidates_metrics.append({
                'index': c,
                'val_score': val_scores[c].item(),
                'loss_decrease': loss_decrease[c].item(),
                'val_select': val_select[c].item(),
            })

        if NGRAM:
            # Unsupervised: rank purely by n-gram loss (ascending), tie-break by training progress.
            candidates_metrics.sort(key=lambda x: (x['val_select'], -x['loss_decrease']))
        else:
            candidates_metrics.sort(key=lambda x: (x['val_score'], x['loss_decrease'], -x['val_select']), reverse=True)

        topk_indices = [m['index'] for m in candidates_metrics[:K_val]]
        active_mask = torch.zeros(C_init, dtype=torch.bool, device=device)
        active_mask[topk_indices] = True

        # Diagnostic: where does the true candidate rank, and would it have survived?
        true_rank = next(i for i, m in enumerate(candidates_metrics) if m['index'] == true_candidate_idx)
        true_select = val_select[true_candidate_idx].item()
        true_acc = val_scores[true_candidate_idx].item()
        survived = "survives" if true_rank < K_val else "PRUNED"
        with torch.no_grad():
            perm_diag = permutation_regularizer(net)
        true_perm = perm_diag[true_candidate_idx].item()
        best_active_perm = perm_diag[active_mask].min().item()
        print(f"   [diag] true candidate rank {true_rank:>5d}/{C_init} ({survived}), "
              f"val_select={true_select:.4f}, MonitorAcc={true_acc:.3f}, "
              f"best_active={val_select[active_mask].min().item():.4f}, "
              f"perm_loss(true)={true_perm:.4f}, perm_loss(best_active)={best_active_perm:.4f}")

        if FORCE_TRUE_ACTIVE:
            active_mask[true_candidate_idx] = True

    if step % LOG_STEP == 0 or step == TOTAL_STEPS - 1:
        active_losses = torch.where(active_mask, loss_per_candidate,
                                    torch.full_like(loss_per_candidate, float('inf')))
        best_idx = torch.argmin(active_losses).item()
        best_candidate_loss = loss_per_candidate[best_idx].item()
        best_candidate_pos = net.get_positions()[best_idx] if net.num_candidates > 1 else net.get_positions()
        best_candidate_int_pos = net.get_integer_positions()[best_idx] if net.num_candidates > 1 else net.get_integer_positions()
        orig_pos = active_original_positions[best_idx]
        monitor = f", MonitorAcc = {val_scores[best_idx].item():.3f}" if NGRAM and step % LOG_STEP == 0 else ""

        print(f"Step {step:>4d} | Active Candidates: {active_mask.sum().item():>5d}/{net.num_candidates} | Best Candidate (Orig: {orig_pos}): Loss = {best_candidate_loss:.4f}, Rounded Pos = {best_candidate_int_pos}, Continuous = {[f'{p:.2f}' for p in best_candidate_pos]}{monitor}")

    if len(val_losses_history) >= 3:
        recent_val = torch.stack(val_losses_history[-3:])
        stds = torch.std(recent_val, dim=0)
        best_idx = torch.argmin(val_losses_history[-1]).item()
        stable = stds[best_idx].item() < STABILITY_EPS
        # In ce mode also require near-zero loss (perfect fit). In ngram mode the loss
        # floor is the English entropy, so stability alone signals convergence.
        converged = stable and (NGRAM or val_losses_history[-1][best_idx].item() < 0.05)
        if converged and not FORCE_TRUE_ACTIVE:
            print(f"\nStep {step:>4d} | Best candidate stabilized. Early stopping.")
            break

print("\nTraining complete.")
val_scores, val_ce, val_ngram = evaluate_validation(net, val_input_indices, val_targets_tensor, loss_fn)
val_select = val_ngram if NGRAM else val_ce
winner_idx = torch.argmin(val_select).item()
best_val_select = val_select[winner_idx].item()
best_val_score = val_scores[winner_idx].item()
winner_pos = net.get_positions()[winner_idx]
winner_int_pos = net.get_integer_positions()[winner_idx]
winner_orig_pos = active_original_positions[winner_idx]

metric_name = "N-gram Loss" if NGRAM else "Loss"
print(f"Winner Candidate (Orig: {winner_orig_pos}): {metric_name} = {best_val_select:.4f}, Monitor Acc = {best_val_score:.4f}, Rounded Pos = {winner_int_pos}, Continuous = {[f'{p:.2f}' for p in winner_pos]}")
print(f"Recovered integer positions: {winner_int_pos}, True: {true_positions}")

print(f"Pruning network down to the winner candidate at index {winner_idx}...")
net.prune_candidates([winner_idx])

print("\nVerification on final test sequence...")
net.eval()
if NGRAM:
    test_plaintext = sample_english(n ** 2)
    target_enigma.reset(true_positions)
    test_cipher = target_enigma.encrypt(test_plaintext)
    learner_decrypted = net.encrypt_string(test_cipher, greedy=True)
    matches = sum(a == b for a, b in zip(test_plaintext, learner_decrypted))
else:
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
