import os
import random
from datetime import datetime
import torch

from enigma_net.fourier.dft_net import EnigmaNet
from enigma_net.fourier.config import dft_config
from comparison.fourier_comparison import compare
from visualiser import visualise_dft

LEARNING_RATE = 0.1
TOTAL_STEPS = 500
TAU_START = 1.0
TAU_END = 0.1
N_TAU_ITERS = TOTAL_STEPS * 0.9

config = dft_config.enigma_config
n = len(config.alphabet)
num_rotors = len(config.rotors)
LEN_STRING = n ** 3

learner = EnigmaNet(
    config,
    load_target=False,
    tau=TAU_START,
    trainable_rotors=dft_config.trainable_rotors,
    trainable_reflector=dft_config.trainable_reflector,
    mapping_type="linear"
)
target = config.build()
optimizer = torch.optim.Adam(learner.parameters(), lr=LEARNING_RATE)
loss_fn = dft_config.loss_fn

for step in range(TOTAL_STEPS):
    tau = TAU_START * (TAU_END / TAU_START) ** (step / N_TAU_ITERS) if step < N_TAU_ITERS else TAU_END
    learner.set_tau(tau)

    positions = [random.randint(0, n - 1) for _ in range(num_rotors)]
    plaintext = "".join(random.choice(config.alphabet) for _ in range(LEN_STRING))

    target.reset(positions)
    learner.reset(positions)
    optimizer.zero_grad()

    outputs = []
    target_labels = []
    for c in plaintext:
        input_vec = torch.zeros(n)
        input_vec[learner.char_to_idx[c]] = 1.0
        target_labels.append(learner.char_to_idx[target.encrypt_char(c)])
        outputs.append(learner(input_vec))

    predictions = torch.stack(outputs)
    targets_tensor = torch.tensor(target_labels, dtype=torch.long)
    loss = loss_fn(predictions, targets_tensor)
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"step {step}, loss {loss.item():.4f}, tau {tau:.4f}")

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
os.makedirs(models_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
weights_path = os.path.join(models_dir, f"dft_learner_{timestamp}.pth")
torch.save(learner.state_dict(), weights_path)

compare(weights_path, config=config)
visualise_dft(learner, config.build(), show_active=False, show_numbers=False)
