import torch
import torch.nn as nn
from .rotor_layer import RotorLayer

class EnigmaNet(nn.Module):
    def __init__(self, config, load_target=False):
        super().__init__()
        self.n = len(config.alphabet)
        self.alphabet = config.alphabet
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

        self.rotors = nn.ModuleList([
            RotorLayer(self.n, self._wiring_to_matrix(r.wiring) if load_target else None)
            for r in config.rotors
        ])
        self.register_buffer("reflector", self._wiring_to_matrix(config.reflector))
        self.notches = [
            r.notch if isinstance(r.notch, int) else config.alphabet.index(r.notch)
            for r in config.rotors
        ]
        self.reset()

        if load_target:
            for p in self.parameters():
                p.requires_grad = False

    def _wiring_to_matrix(self, wiring):
        matrix = torch.zeros(self.n, self.n)
        for col, char in enumerate(wiring):
            matrix[self.char_to_idx[char], col] = 1.0
        return matrix

    def step(self):
        for i in range(len(self.rotors) - 1, -1, -1):
            at_notch = self.positions[i] == self.notches[i]
            self.positions[i] = (self.positions[i] + 1) % self.n
            if not at_notch:
                break

    def forward(self, v):
        self.step()
        for r, pos in zip(reversed(self.rotors), reversed(self.positions)):
            v = r(v, pos)
        v = self.reflector @ v
        for r, pos in zip(self.rotors, self.positions):
            v = r.backward_pass(v, pos)
        return torch.softmax(v, dim=-1)

    def reset(self, positions=None):
        if positions is None:
            positions = [0] * len(self.rotors)
        self.positions = [
            self.alphabet.index(p) if isinstance(p, str) else int(p)
            for p in positions
        ]

    def encrypt_string(self, text):
        res = []
        for c in text:
            if c not in self.char_to_idx:
                res.append(c)
                continue
            v = torch.zeros(self.n)
            v[self.char_to_idx[c]] = 1.0
            out = self.forward(v)
            res.append(self.alphabet[torch.argmax(out).item()])
        return "".join(res)

