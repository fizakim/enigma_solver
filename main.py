from enigma.config.config3 import build

positions = [0, 0, 0]
plaintext = "ABC"
print("Plaintext:  ", plaintext)

machine_enc = build(positions)

ciphertext = machine_enc.encrypt(plaintext)
print("Ciphertext:", ciphertext)

machine_dec = build(positions)
decrypted = machine_dec.decrypt(ciphertext)
print("Decrypted: ", decrypted)

