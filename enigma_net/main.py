import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config3 import config3
from enigma_net.enigma_net import EnigmaNet

net = EnigmaNet(config3, load_target=False)
target_sim = config3.build()


plaintext = "ABC"
print(f"Random net:  '{plaintext}' -> '{net.encrypt_string(plaintext)}'")
print(f"Target sim:  '{plaintext}' -> '{target_sim.encrypt(plaintext)}'")

