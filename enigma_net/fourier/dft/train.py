import sys
import os
import random
from datetime import datetime
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from enigma_net.fourier.dft.net import EnigmaNet
from comparison.fourier_comparison import compare
from visualiser import visualise_fourier

from enigma_net import CrossEntropyLoss
from enigma_net.train_config import TrainConfig
from config.alphabet3 import alphabet3
from config.alphabet5 import alphabet5
from config.alphabet10 import alphabet10



train_config = TrainConfig(
    enigma_config=alphabet3,
    loss_fn=CrossEntropyLoss(),
    trainable_rotors=None,
    trainable_reflector=False,
)

LEARNING_RATE = 0.1
TOTAL_STEPS = 500
LOG_STEP = 10
TAU_START = 1.0
TAU_END = 0.1
N_TAU_ITERS = TOTAL_STEPS * 0.9
LEN_STRING = len(train_config.enigma_config.alphabet) ** 3

learner = EnigmaNet(
    train_config.enigma_config, 
    load_target=False, 
    tau=TAU_START, 
    trainable_rotors=train_config.trainable_rotors,
    trainable_reflector=train_config.trainable_reflector,
    mapping_type="linear"
)
target = train_config.enigma_config.build()
optimizer = torch.optim.Adam(learner.parameters(), lr=LEARNING_RATE)
loss_fn = train_config.loss_fn

print("Training (Fourier, supervised)...")
n_alphabet = len(train_config.enigma_config.alphabet)
n_rotors = len(train_config.enigma_config.rotors)

for step in range(TOTAL_STEPS):
    tau = TAU_START * (TAU_END / TAU_START) ** (step / N_TAU_ITERS) if step < N_TAU_ITERS else TAU_END
    learner.set_tau(tau)
    
    positions = [random.randint(0, n_alphabet - 1) for _ in range(n_rotors)]
    plaintext = "".join(random.choice(train_config.enigma_config.alphabet) for _ in range(LEN_STRING))
    
    target.reset(positions)
    learner.reset(positions)
    optimizer.zero_grad()
    
    outputs = []
    target_labels = []
    for c in plaintext:
        input_vec = torch.zeros(n_alphabet)
        input_vec[learner.char_to_idx[c]] = 1.0
        
        target_labels.append(learner.char_to_idx[target.encrypt_char(c)])
        outputs.append(learner(input_vec))
        
    predictions = torch.stack(outputs)
    targets = torch.tensor(target_labels, dtype=torch.long)
    ce_loss = loss_fn(predictions, targets)
    
    ce_loss.backward()
    optimizer.step()
    
    if step % LOG_STEP == 0:
        print(f"step {step}, loss {ce_loss.item():.4f}, tau {tau:.4f}")

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
os.makedirs(models_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
weights_path = os.path.join(models_dir, f"dft_learner_{timestamp}.pth")
torch.save(learner.state_dict(), weights_path)
print(f"Saved trained learner weights to '{weights_path}'")

print("\nRunning compare.py evaluation...")
compare(weights_path, config=train_config.enigma_config)

visualise_fourier(learner, train_config.enigma_config.build(), show_active=False, show_numbers=False)
