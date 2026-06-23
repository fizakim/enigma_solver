import sys
import os
import random
from datetime import datetime
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from enigma_net.fourier.q_net.net import QNet
from comparison.fourier_comparison import compare
from visualiser import visualise_q_net

from enigma_net import CrossEntropyLoss
from enigma_net.train_config import TrainConfig
from config.alphabet3 import alphabet3
from config.alphabet5 import alphabet5
from config.alphabet10 import alphabet10
from config.alphabet15 import alphabet15
from config.alphabet26 import alphabet26

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LOAD_TARGET = False

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
    loss_fn = train_config.loss_fn

    print("Training QNet (batched)...")
    n_alphabet = len(train_config.enigma_config.alphabet)
    n_rotors = len(train_config.enigma_config.rotors)

    for step in range(TOTAL_STEPS):
        positions = [random.randint(0, n_alphabet - 1) for _ in range(n_rotors)]
        plaintext = [random.choice(train_config.enigma_config.alphabet) for _ in range(LEN_STRING)]

        target = train_config.enigma_config.build()
        target.reset(positions)
        learner.reset(positions)
        optimizer.zero_grad()

        # Build input indices and target labels in one Python pass (no autograd needed)
        input_indices = [learner.char_to_idx[c] for c in plaintext]
        target_labels = [learner.char_to_idx[target.encrypt_char(c)] for c in plaintext]

        # Batched forward: [T, n] logits in a single vectorised pass
        predictions = learner.encrypt_sequence(input_indices)
        targets = torch.tensor(target_labels, dtype=torch.long, device=device)
        ce_loss = loss_fn(predictions, targets)

        ce_loss.backward()
        optimizer.step()

        if step % LOG_STEP == 0:
            print(f"step {step:>4d}, loss {ce_loss.item():.4f}")

    torch.save(learner.state_dict(), weights_path)
    print(f"Saved weights to '{weights_path}'")
    compare(weights_path, config=train_config.enigma_config)
    visualise_q_net(learner, train_config.enigma_config.build(), show_numbers=False)
