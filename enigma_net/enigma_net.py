import torch
import torch.nn as nn
from .rotor_layer import RotorLayer

class EnigmaNet(nn.Module):
    def __init__(self, config, load_target=False, tau=0.1, iterations=10):
        super().__init__()
        self.config = config
        self.n = len(config.alphabet)
        self.alphabet = config.alphabet
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

        self.rotors = nn.ModuleList([
            RotorLayer(
                self.n, 
                target_wiring=torch.from_numpy(config.wiring_to_matrix(r.wiring)).float() if load_target else None,
                tau=tau,
                iterations=iterations
            )
            for r in config.rotors
        ])
        self.register_buffer("reflector", torch.from_numpy(config.wiring_to_matrix(config.reflector)).float())
        self.notches = [config.parse_position(r.notch) for r in config.rotors]
        self.reset()

        if load_target:
            for p in self.parameters():
                p.requires_grad = False

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
        return v

    def reset(self, positions=None):
        if positions is None:
            positions = [0] * len(self.rotors)
        self.positions = [self.config.parse_position(p) for p in positions]

    def encrypt_string(self, text):
        res = []
        for c in text:
            if c not in self.char_to_idx:
                res.append(c)
                continue
            v = torch.zeros(self.n)
            v[self.char_to_idx[c]] = 1.0
            out = torch.softmax(self.forward(v), dim=-1)
            res.append(self.alphabet[torch.argmax(out).item()])
        return "".join(res)

