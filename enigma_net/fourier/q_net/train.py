import sys
import os
import glob
import random
from datetime import datetime
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from enigma_net.fourier.q_net.net import QNet
from comparison.fourier_comparison import compare
from visualiser import visualise_q_net

from enigma_net import CrossEntropyLoss, NgramLoss, load_ngram_logprobs
from enigma_net.train_config import TrainConfig
from transformer.loss import load_transformer_lm, TransformerLoss
from config.alphabet3 import alphabet3
from config.alphabet5 import alphabet5
from config.alphabet10 import alphabet10
from config.alphabet15 import alphabet15
from config.alphabet26 import alphabet26

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LOAD_TARGET = False

# "ce":          supervised — feed plaintext and match the target's ciphertext.
# "ngram":       unsupervised — score the learner's decryption with a trigram prior.
# "transformer": unsupervised — score with a frozen char-level transformer LM.
LOSS_MODE = "transformer"   # "ce", "ngram", or "transformer"
UNSUPERVISED = LOSS_MODE in ("ngram", "transformer")
TAU = 0.5             # softmax temperature for the soft decoding

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NGRAM_PATH = os.path.join(_ROOT, "language", "ngram", "3grams.pth")
CORPUS_PATH = os.path.join(_ROOT, "language", "fineweb", "fineweb.txt")
MODELS_DIR = os.path.join(_ROOT, "models")

train_config = TrainConfig(
    enigma_config=alphabet26,
    loss_fn=CrossEntropyLoss(),
    trainable_rotors=None,
    trainable_reflector=False,
)

LEARNING_RATE = 0.01
TOTAL_STEPS = 250
LOG_STEP = 10
LEN_STRING = len(train_config.enigma_config.alphabet) ** 3

learner = QNet(
    train_config.enigma_config,
    load_target=LOAD_TARGET,
    trainable_rotors=train_config.trainable_rotors,
    trainable_reflector=train_config.trainable_reflector,
).to(device)

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
os.makedirs(models_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
weights_path = os.path.join(models_dir, f"q_net_learner_{timestamp}.pth")

if LOAD_TARGET:
    print("Verifying mathematical correctness...")
    torch.save(learner.state_dict(), weights_path)
    compare(weights_path, config=train_config.enigma_config)
else:
    optimizer = torch.optim.Adam(learner.parameters(), lr=LEARNING_RATE)
    n_alphabet = len(train_config.enigma_config.alphabet)
    n_rotors = len(train_config.enigma_config.rotors)

    if LOSS_MODE == "ngram":
        loss_fn = NgramLoss(load_ngram_logprobs(NGRAM_PATH, n_alphabet, device), tau=TAU).to(device)
    elif LOSS_MODE == "transformer":
        ckpt_paths = sorted(glob.glob(os.path.join(MODELS_DIR, "transformer_lm_*.pth")))
        if not ckpt_paths:
            raise FileNotFoundError(
                f"No transformer_lm_*.pth found in {MODELS_DIR}. Run transformer/train.py first."
            )
        print(f"Loading transformer LM: {ckpt_paths[-1]}")
        _lm = load_transformer_lm(ckpt_paths[-1], device)
        loss_fn = TransformerLoss(_lm, tau=TAU)
    else:
        loss_fn = train_config.loss_fn

    # Unsupervised modes need real English plaintext so a correct decryption scores high.
    if UNSUPERVISED:
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            corpus = "".join(ch for ch in f.read() if ch in learner.char_to_idx)
        if len(corpus) < 2 * LEN_STRING:
            raise RuntimeError(f"Corpus too small ({len(corpus)} chars) for LEN_STRING={LEN_STRING}.")

        def sample_english(length):
            start = random.randint(0, len(corpus) - length - 1)
            return corpus[start:start + length]

        # Positions are known and fixed; the learner recovers the wiring.
        true_positions = [random.randint(0, n_alphabet - 1) for _ in range(n_rotors)]
        print(f"Unsupervised ({LOSS_MODE}) mode. Known start positions: {true_positions}")

    print(f"Training QNet (batched), loss mode = {LOSS_MODE}...")

    for step in range(TOTAL_STEPS):
        target = train_config.enigma_config.build()
        optimizer.zero_grad()

        if UNSUPERVISED:
            # Encrypt real English, feed the ciphertext, score the decryption.
            # monitor_labels are the plaintext, used only for the readout.
            plaintext = sample_english(LEN_STRING)
            target.reset(true_positions)
            learner.reset(true_positions)

            input_indices = [learner.char_to_idx[target.encrypt_char(c)] for c in plaintext]
            monitor_labels = torch.tensor(
                [learner.char_to_idx[c] for c in plaintext], dtype=torch.long, device=device
            )

            predictions = learner.encrypt_sequence(input_indices)        # [T, n]
            loss = loss_fn(predictions.unsqueeze(0)).mean()
        else:
            positions = [random.randint(0, n_alphabet - 1) for _ in range(n_rotors)]
            plaintext = [random.choice(train_config.enigma_config.alphabet) for _ in range(LEN_STRING)]
            target.reset(positions)
            learner.reset(positions)

            input_indices = [learner.char_to_idx[c] for c in plaintext]
            target_labels = [learner.char_to_idx[target.encrypt_char(c)] for c in plaintext]

            predictions = learner.encrypt_sequence(input_indices)        # [T, n]
            targets = torch.tensor(target_labels, dtype=torch.long, device=device)
            loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        if step % LOG_STEP == 0:
            if UNSUPERVISED:
                with torch.no_grad():
                    acc = (predictions.argmax(dim=-1) == monitor_labels).float().mean().item()
                print(f"step {step:>4d}, {LOSS_MODE} loss {loss.item():.4f}, monitor acc {acc:.3f}")
            else:
                print(f"step {step:>4d}, loss {loss.item():.4f}")

    torch.save(learner.state_dict(), weights_path)
    print(f"Saved weights to '{weights_path}'")
    compare(weights_path, config=train_config.enigma_config)
    visualise_q_net(learner, train_config.enigma_config.build(), show_numbers=False)
