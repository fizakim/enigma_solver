import os
import glob
import random
from datetime import datetime
import torch

from enigma_net.fourier.q_net import QNet
from enigma_net.fourier.config import alphabet26_config
from enigma_net import NgramLoss, load_ngram_logprobs
from transformer.loss import load_transformer_lm, TransformerLoss
from enigma_net.ce_approximator.model import load_ce_approximator
from comparison.fourier_comparison import compare
from visualiser import visualise_q_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_MODE = "ce_approximator"
LEARNING_RATE = 0.001
TOTAL_STEPS = 2500
TAU = 0.5

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NGRAM_PATH = os.path.join(_ROOT, "language", "ngram", "3grams.pth")
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")
MODELS_DIR = os.path.join(_ROOT, "models")
CE_APPROX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ce_approximator", "models")
)

config = alphabet26_config.enigma_config
n = len(config.alphabet)
num_rotors = len(config.rotors)
LEN_STRING = n ** 3

learner = QNet(
    config,
    load_target=False,
    trainable_rotors=alphabet26_config.trainable_rotors,
    trainable_reflector=alphabet26_config.trainable_reflector,
).to(device)

UNSUPERVISED = LOSS_MODE in ("ngram", "transformer", "ce_approximator")

if LOSS_MODE == "ngram":
    loss_fn = NgramLoss(load_ngram_logprobs(NGRAM_PATH, n, device), tau=TAU).to(device)
elif LOSS_MODE == "transformer":
    ckpt_paths = sorted(glob.glob(os.path.join(MODELS_DIR, "transformer_lm_*.pth")))
    loss_fn = TransformerLoss(load_transformer_lm(ckpt_paths[-1], device), tau=TAU)
elif LOSS_MODE == "ce_approximator":
    _ce_ckpts = sorted(glob.glob(os.path.join(CE_APPROX_DIR, "ce_approximator_*.pth")))
    loss_fn = load_ce_approximator(_ce_ckpts[-1], device=str(device))
else:
    loss_fn = alphabet26_config.loss_fn

if UNSUPERVISED:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = "".join(ch for ch in f.read() if ch in learner.char_to_idx)

    def sample_english(length):
        start = random.randint(0, len(corpus) - length - 1)
        return corpus[start:start + length]

    true_positions = [random.randint(0, n - 1) for _ in range(num_rotors)]

optimizer = torch.optim.Adam(learner.parameters(), lr=LEARNING_RATE)

for step in range(TOTAL_STEPS):
    target = config.build()
    optimizer.zero_grad()

    if UNSUPERVISED:
        plaintext = sample_english(LEN_STRING)
        target.reset(true_positions)
        learner.reset(true_positions)

        input_indices = [learner.char_to_idx[target.encrypt_char(c)] for c in plaintext]
        monitor_labels = torch.tensor(
            [learner.char_to_idx[c] for c in plaintext], dtype=torch.long, device=device
        )

        predictions = learner.encrypt_sequence(input_indices)
        if LOSS_MODE == "ce_approximator":
            cipher_t = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)
            positions_t = learner.step_positions(len(input_indices)).unsqueeze(0)
            state_t = learner.state_features().unsqueeze(0)
            loss = loss_fn(
                predictions.unsqueeze(0),
                cipher=cipher_t, positions=positions_t, qnet_state=state_t,
            ).mean()
        else:
            loss = loss_fn(predictions.unsqueeze(0)).mean()
    else:
        positions = [random.randint(0, n - 1) for _ in range(num_rotors)]
        plaintext = [random.choice(config.alphabet) for _ in range(LEN_STRING)]
        target.reset(positions)
        learner.reset(positions)

        input_indices = [learner.char_to_idx[c] for c in plaintext]
        target_labels = [learner.char_to_idx[target.encrypt_char(c)] for c in plaintext]

        predictions = learner.encrypt_sequence(input_indices)
        targets_tensor = torch.tensor(target_labels, dtype=torch.long, device=device)
        loss = loss_fn(predictions, targets_tensor)

    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        if UNSUPERVISED:
            with torch.no_grad():
                acc = (predictions.argmax(dim=-1) == monitor_labels).float().mean().item()
            print(f"step {step:>4d}, {LOSS_MODE} loss {loss.item():.4f}, monitor acc {acc:.3f}")
        else:
            print(f"step {step:>4d}, loss {loss.item():.4f}")

os.makedirs(MODELS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
weights_path = os.path.join(MODELS_DIR, f"q_net_learner_{timestamp}.pth")
torch.save(learner.state_dict(), weights_path)

compare(weights_path, config=config)
visualise_q_net(learner, config.build(), show_numbers=False)
