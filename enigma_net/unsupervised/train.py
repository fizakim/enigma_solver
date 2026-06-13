import sys
import os
import random
from datetime import datetime
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from enigma_net.enigma_net import EnigmaNet
from enigma_net.compare import compare
from visualiser import visualise

from enigma_net import NgramLoss
from enigma_net.train_config import TrainConfig
from n_gram.generator import load_ngram_counts
from config.alphabet3 import alphabet3
from config.alphabet5 import alphabet5
from config.alphabet10 import alphabet10
from config.alphabet15 import alphabet15
from config.alphabet26 import alphabet26

NGRAM_COUNTS_PATH = "n_gram/counts3.pth"

train_config = TrainConfig(
    enigma_config=alphabet3,
    loss_fn=NgramLoss(load_ngram_counts(NGRAM_COUNTS_PATH)),
    trainable_rotors=None,
    trainable_reflector=True,
)


LEARNING_RATE = 0.1
TOTAL_STEPS = 100000
LOG_STEP = 100
TAU_START = 1.0
TAU_END = 0.1
N_TAU_ITERS = TOTAL_STEPS * 0.9
ITERATIONS = 10
OPTIMIZER_CLASS = torch.optim.Adam
LEN_STRING = 3 ** 3


learner = EnigmaNet(
    train_config.enigma_config,
    load_target=False,
    tau=TAU_START,
    iterations=ITERATIONS,
    trainable_rotors=train_config.trainable_rotors,
    trainable_reflector= False#train_config.trainable_reflector
)

optimizer = OPTIMIZER_CLASS(learner.parameters(), lr=LEARNING_RATE)
loss_fn = train_config.loss_fn

print("Training (unsupervised)...")
tau = TAU_START
n_alphabet = len(train_config.enigma_config.alphabet)
n_rotors = len(train_config.enigma_config.rotors)

for step in range(TOTAL_STEPS):
    if step % 100 == 0:
        if step < N_TAU_ITERS:
            tau = TAU_START * (TAU_END / TAU_START) ** (step / N_TAU_ITERS)
        else:
            tau = TAU_END
        learner.set_tau(tau)

    positions = [random.randint(0, n_alphabet - 1) for _ in range(n_rotors)]
    plaintext = "".join(random.choice(train_config.enigma_config.alphabet) for _ in range(LEN_STRING))

    learner.reset(positions)
    optimizer.zero_grad()

    outputs = []
    for c in plaintext:
        input_vec = torch.zeros(n_alphabet)
        input_vec[learner.char_to_idx[c]] = 1.0
        learner_out = learner(input_vec)
        outputs.append(learner_out)

    predictions = torch.stack(outputs)
    total_loss = loss_fn(predictions)

    total_loss.backward()
    optimizer.step()

    if step % LOG_STEP == 0:
        print(f"step {step}, loss {total_loss.item():.6f}, tau {tau:.4f}")

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(models_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
weights_path = os.path.join(models_dir, f"learner_{timestamp}.pth")
torch.save(learner.state_dict(), weights_path)
print(f"Saved trained learner weights to '{weights_path}'")

print("\nRunning compare.py evaluation...")
compare(weights_path, config=train_config.enigma_config)

visualise(learner, train_config.enigma_config.build(), show_active=False, show_numbers=False)
