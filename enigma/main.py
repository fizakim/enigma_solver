import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.alphabet3 import alphabet3

positions = [0, 0, 0]
plaintext = alphabet3.language.generate_sentence(3) if alphabet3.language else "ABC"
print("Plaintext:  ", plaintext)

machine_enc = alphabet3.build(positions)

ciphertext = machine_enc.encrypt(plaintext)
print("Ciphertext:", ciphertext)

machine_dec = alphabet3.build(positions)
decrypted = machine_dec.decrypt(ciphertext)
print("Decrypted: ", decrypted)
