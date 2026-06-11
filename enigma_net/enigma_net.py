import torch
import torch.nn as nn
from .rotor_layer import RotorLayer

class EnigmaNet(nn.Module):
    def __init__(self, config, load_target=False, tau=0.1, iterations=10, trainable_rotors=None, trainable_reflector=False):
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
        
        if trainable_reflector and not load_target:
            self.reflector_logits = nn.Parameter(torch.randn(self.n, self.n))
            from .sinkhorn import Sinkhorn
            self.reflector_sinkhorn = Sinkhorn(tau, iterations)
        else:
            self.reflector_logits = None
            self.register_buffer("_reflector", torch.from_numpy(config.wiring_to_matrix(config.reflector)).float())

        self.notches = [config.parse_position(r.notch) for r in config.rotors]
        self.reset()

        if load_target:
            for p in self.parameters():
                p.requires_grad = False
        elif trainable_rotors is not None:
            for i, r in enumerate(self.rotors):
                if i not in trainable_rotors and r.logits is not None:
                    r.logits.requires_grad = False

    @property
    def reflector(self):
        if self.reflector_logits is not None:
            sym_logits = (self.reflector_logits + self.reflector_logits.T) / 2.0
            w = self.reflector_sinkhorn(sym_logits)
            return (w + w.T) / 2.0
        return self._reflector

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        reflector_key = prefix + "reflector"
        _reflector_key = prefix + "_reflector"
        if reflector_key in state_dict and _reflector_key not in state_dict:
            state_dict[_reflector_key] = state_dict.pop(reflector_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

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

    def set_tau(self, tau):
        for r in self.rotors:
            if r.sinkhorn is not None:
                r.sinkhorn.tau = tau


    def encrypt_string(self, text, greedy=False):
        res = []
        for c in text:
            if c not in self.char_to_idx:
                res.append(c)
                continue
            v = torch.zeros(self.n)
            v[self.char_to_idx[c]] = 1.0
            out = torch.softmax(self.forward(v), dim=-1)
            if greedy:
                idx = torch.argmax(out).item()
            else:
                idx = torch.multinomial(out, 1).item()
            res.append(self.alphabet[idx])
        return "".join(res)

