import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.alphabet3 import alphabet3

positions = [0, 0, 0]
import random
plaintext = "".join(random.choice(alphabet3.alphabet) for _ in range(3))
print("Plaintext:  ", plaintext)

machine_enc = alphabet3.build(positions)

ciphertext = machine_enc.encrypt(plaintext)
print("Ciphertext:", ciphertext)

machine_dec = alphabet3.build(positions)
decrypted = machine_dec.decrypt(ciphertext)
print("Decrypted: ", decrypted)
