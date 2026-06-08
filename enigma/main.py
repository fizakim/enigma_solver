import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config3 import config3

positions = [0, 0, 0]
plaintext = "ABC"
print("Plaintext:  ", plaintext)

machine_enc = config3.build(positions)

ciphertext = machine_enc.encrypt(plaintext)
print("Ciphertext:", ciphertext)

machine_dec = config3.build(positions)
decrypted = machine_dec.decrypt(ciphertext)
print("Decrypted: ", decrypted)
