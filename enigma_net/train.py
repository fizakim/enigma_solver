import sys
import os
import random
from datetime import datetime
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config3 import config3
from enigma_net.enigma_net import EnigmaNet
from enigma_net.compare import compare
from visualiser import visualise


tau_start = 1
tau_end = 0.1
total_steps = 500
n_tau_iters = total_steps*0.9
log_step = 100
len_string = 27


learner = EnigmaNet(config3, load_target=False, tau=tau_start, iterations=10)
target = config3.build()

optimizer = torch.optim.Adam(learner.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print("Training...")
tau = tau_start
for step in range(total_steps):
    if step % 100 == 0:
        if step < n_tau_iters:
            tau = tau_start * (tau_end / tau_start) ** (step / n_tau_iters)
        else:
            tau = tau_end
        learner.set_tau(tau)
    
    positions = [random.randint(0, 2) for i in range(3)]
    plaintext = "".join(random.choice("ABC") for _ in range(len_string))
    
    target.reset(positions)
    learner.reset(positions)
    optimizer.zero_grad()
    
    total_loss = 0.0
    for c in plaintext:
        input_vec = torch.zeros(3)
        input_vec[learner.char_to_idx[c]] = 1.0
        
        target_char = target.encrypt_char(c)
        target_label = torch.tensor(learner.char_to_idx[target_char], dtype=torch.long)
        
        learner_out = learner(input_vec)
        total_loss = total_loss + loss_fn(learner_out.unsqueeze(0), target_label.unsqueeze(0))
    
    total_loss.backward()
    optimizer.step()
    
    if step % log_step  == 0:
        print(f"step {step}, loss {total_loss.item():.4f}, tau {tau:.4f}")

models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
weights_path = os.path.join(models_dir, f"learner_{timestamp}.pth")
torch.save(learner.state_dict(), weights_path)
print(f"Saved trained learner weights to '{weights_path}'")

print("\nRunning compare.py evaluation...")

compare(weights_path)

visualise(learner, config3.build(), show_active=False)
