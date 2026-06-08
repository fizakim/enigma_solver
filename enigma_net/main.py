import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config3 import config3
from enigma_net.enigma_net import EnigmaNet
from visualiser import visualise

net = EnigmaNet(config3, load_target=False)
target_sim = config3.build()

plaintext = config3.language.generate_sentence(3) if config3.language else "ABC"

print(f"Random net:  '{plaintext}' -> '{net.encrypt_string(plaintext)}'")
print(f"Target sim:  '{plaintext}' -> '{target_sim.encrypt(plaintext)}'")

visualise(net, target_sim, show_active=True)



