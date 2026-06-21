import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.alphabet3 import alphabet3
from enigma_net.enigma_net import EnigmaNet
from visualiser import visualise

net = EnigmaNet(alphabet3, load_target=False)
target_sim = alphabet3.build()

import random
plaintext = "".join(random.choice(alphabet3.alphabet) for _ in range(3))

print(f"Random net:  '{plaintext}' -> '{net.encrypt_string(plaintext)}'")
print(f"Target sim:  '{plaintext}' -> '{target_sim.encrypt(plaintext)}'")

visualise(net, target_sim, show_active=True)



