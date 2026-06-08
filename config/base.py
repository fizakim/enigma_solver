import numpy as np
from enigma import Enigma, Plugboard, Reflector, Rotor

class RotorConfig:
    def __init__(self, wiring, notch):
        self.wiring = wiring
        self.notch = notch

class EnigmaConfig:
    def __init__(self, alphabet, rotors, reflector, plugboard_pairs=None, language=None):
        self.alphabet = alphabet
        self.rotors = rotors
        self.reflector = reflector
        self.plugboard_pairs = plugboard_pairs or []
        self.language = language

    def wiring_to_matrix(self, wiring):
        n = len(self.alphabet)
        idx = {c: i for i, c in enumerate(self.alphabet)}
        matrix = np.zeros((n, n), dtype=int)
        for col, char in enumerate(wiring):
            matrix[idx[char], col] = 1
        return matrix

    def parse_position(self, pos):
        return self.alphabet.index(pos) if isinstance(pos, str) else int(pos)

    def build(self, positions=None):
        n = len(self.alphabet)
        if positions is None:
            positions = [0] * len(self.rotors)
        
        pos_indices = [self.parse_position(p) for p in positions]

        rotor_objs = []
        for r, pos in zip(self.rotors, pos_indices):
            notch = self.parse_position(r.notch)
            rotor_objs.append(Rotor(self.wiring_to_matrix(r.wiring), notch, pos))

        reflector_obj = Reflector(self.wiring_to_matrix(self.reflector))

        plugboard_matrix = np.eye(n, dtype=int)
        for pair in self.plugboard_pairs:
            i, j = self.alphabet.index(pair[0]), self.alphabet.index(pair[1])
            plugboard_matrix[[i, j]] = plugboard_matrix[[j, i]]

        return Enigma(rotor_objs, reflector_obj, Plugboard(plugboard_matrix, n), self.alphabet)
