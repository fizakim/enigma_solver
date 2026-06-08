import numpy as np
from .plugboard import Plugboard

class Enigma:
    def __init__(self, rotors, reflector, plugboard=None, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        self.alphabet = alphabet
        self.n = len(alphabet)
        self.char_to_idx = {c: i for i, c in enumerate(alphabet)}
        self.rotors = rotors
        self.reflector = reflector
        self.plugboard = plugboard or Plugboard(size=self.n)

    def encrypt_char(self, c):
        if c not in self.char_to_idx:
            return c
        for r in reversed(self.rotors):
            if not r.step():
                break
        v = np.eye(self.n, dtype=int)[self.char_to_idx[c]]
        v = self.plugboard.swap(v)
        for r in reversed(self.rotors):
            v = r.forward(v)
        v = self.reflector.reflect(v)
        for r in self.rotors:
            v = r.backward(v)
        v = self.plugboard.swap(v)
        return self.alphabet[np.argmax(v)]

    def encrypt(self, text):
        return "".join(self.encrypt_char(c) for c in text)

    decrypt = encrypt

    def reset(self, positions):
        for r, pos in zip(self.rotors, positions):
            r.position = pos
