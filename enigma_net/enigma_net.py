import torch
import torch.nn as nn
from .rotor_layer import RotorLayer

class EnigmaNet(nn.Module):
    def __init__(self, config, load_target=False, sinkhorn=None):
        super().__init__()
        self.config = config
        self.n = len(config.alphabet)
        self.alphabet = config.alphabet
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.sinkhorn = sinkhorn

        self.rotors = nn.ModuleList([
            RotorLayer(self.n, torch.from_numpy(config.wiring_to_matrix(r.wiring)).float() if load_target else None)
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
            w = self.sinkhorn(r.wiring) if self.sinkhorn is not None else r.wiring
            v = r(v, pos, wiring=w)
        v = self.reflector @ v
        for r, pos in zip(self.rotors, self.positions):
            w = self.sinkhorn(r.wiring) if self.sinkhorn is not None else r.wiring
            v = r.backward_pass(v, pos, wiring=w)
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

