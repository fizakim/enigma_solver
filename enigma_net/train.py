import sys
import os
import random
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config3 import config3
from enigma_net.enigma_net import EnigmaNet
from visualiser import visualise

tau_start = 2
tau_end = 0.1
n_tau_iters = 80_000
total_steps = 100_000

learner = EnigmaNet(config3, load_target=False, tau=tau_start, iterations=10)
target = EnigmaNet(config3, load_target=True)

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
    
    positions = [random.randint(0, 1) for i in range(3)]
    char_idx = random.randint(0, 2)
    
    target.reset(positions)
    input_vec = torch.zeros(3)
    input_vec[char_idx] = 1.0
    with torch.no_grad():
        target_out = target(input_vec)
    target_label = torch.argmax(target_out)
    
    learner.reset(positions)
    optimizer.zero_grad()
    learner_out = learner(input_vec)
    loss = loss_fn(learner_out.unsqueeze(0), target_label.unsqueeze(0))
    loss.backward()
    optimizer.step()
    
    if step % 1000 == 0:
        print(f"step {step}, loss {loss.item():.4f}, tau {tau:.4f}")

print("\nValidation:")
correct = 0
for i in range(10):
    positions = [random.randint(0, 1) for _ in range(3)]
    plaintext = "".join(random.choice("ABC") for _ in range(5))
    
    learner.reset(positions)
    target.reset(positions)
    
    learner_out = learner.encrypt_string(plaintext)
    target_out = target.encrypt_string(plaintext)
    if learner_out == target_out:
        correct += 1
    print(f"<{learner_out == target_out}> pos={positions} input='{plaintext}' learner='{learner_out}' target='{target_out}'")

print(f"Accuracy: {correct}/10")

visualise(learner, config3.build(), show_active=False)
