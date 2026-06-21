import math
import itertools
import torch
import torch.nn as nn
from ..mappings import get_mapping
from ..reflector import Reflector

def get_logits(n, perm):
    spectrum = torch.fft.rfft(perm)
    spectral = spectrum[1:]
    return torch.cat([spectral.real, spectral.imag if n % 2 != 0 else spectral.imag[:-1]])

class QNet(nn.Module):
    def __init__(self, config, load_target=False, tau=0.1, trainable_reflector=False, mapping_type="softmax"):
        super().__init__()
        self.config = config
        self.n = len(config.alphabet)
        self.alphabet = config.alphabet
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.tau = tau
        self.dc = self.n * (self.n - 1) / 2.0
        self.mapping = get_mapping(mapping_type, self.n)
        self.num_rotors = len(config.rotors)
        self.num_positions = self.n ** self.num_rotors

        if load_target:
            target = config.build()
            target_wiring = [torch.from_numpy(r.matrix).float() for r in target.rotors]
            all_positions = list(itertools.product(range(self.n), repeat=self.num_rotors))
            
            logits_list = []
            for pos in all_positions:
                M_fwd = torch.eye(self.n)
                for W, r_pos in zip(target_wiring, pos):
                    M_fwd = M_fwd @ torch.roll(W, shifts=(-r_pos, -r_pos), dims=(0, 1))
                perm = M_fwd.argmax(dim=0).float()
                logits_list.append(get_logits(self.n, perm))
                
            self.register_buffer('logits', torch.stack(logits_list))
        else:
            logits_list = [get_logits(self.n, torch.randperm(self.n).float()) for _ in range(self.num_positions)]
            self.logits = nn.Parameter(torch.stack(logits_list))

        if trainable_reflector and not load_target:
            self.reflector_layer = Reflector(self.n, tau=tau, mapping_type=mapping_type)
        else:
            self.reflector_layer = Reflector(
                self.n,
                target_reflector=torch.from_numpy(config.wiring_to_matrix(config.reflector)).float(),
                tau=tau,
                mapping_type=mapping_type
            )

        self.notches = [config.parse_position(r.notch) for r in config.rotors]
        self.reset()

        if load_target:
            for p in self.parameters():
                p.requires_grad = False

    @property
    def reflector(self):
        return self.reflector_layer.get_matrix()

    def _positions_to_index(self, positions):
        idx = 0
        for p in positions:
            idx = idx * self.n + self.config.parse_position(p)
        return idx

    def step(self):
        for i in range(len(self.notches) - 1, -1, -1):
            at_notch = self.positions[i] == self.notches[i]
            self.positions[i] = (self.positions[i] + 1) % self.n
            if not at_notch:
                break

    def get_permutation_values(self, flat_idx):
        logits = self.logits[flat_idx]
        half = self.n // 2
        real = logits[:half]
        imag = logits[half:]
        if self.n % 2 == 0:
            imag = torch.cat([imag, torch.zeros(1, device=logits.device)])
        spectral = torch.complex(real, imag)
        
        dc_tensor = torch.tensor([self.dc], dtype=spectral.dtype, device=spectral.device)
        rfft_spectrum = torch.cat([dc_tensor, spectral])
        return torch.fft.irfft(rfft_spectrum, n=self.n)

    def get_Q(self, positions):
        flat_idx = self._positions_to_index(positions)
        perm_values = self.get_permutation_values(flat_idx)
        return self.mapping(perm_values, self.tau)

    def forward(self, v):
        self.step()
        Q = self.get_Q(self.positions)
        v = Q @ v
        v = self.reflector_layer(v)
        v = Q.T @ v
        return v

    def forward_matrix(self, positions):
        self.reset(positions)
        self.step()
        Q = self.get_Q(self.positions)
        return Q.T @ self.reflector @ Q

    def reset(self, positions=None):
        if positions is None:
            positions = [0] * self.num_rotors
        self.positions = [self.config.parse_position(p) for p in positions]

    def set_tau(self, tau):
        self.tau = tau
        if hasattr(self, "reflector_layer") and self.reflector_layer is not None:
            self.reflector_layer.tau = tau

    def encrypt_string(self, text, greedy=False):
        res = []
        for c in text:
            if c not in self.char_to_idx:
                res.append(c)
                continue
            v = torch.zeros(self.n)
            v[self.char_to_idx[c]] = 1.0
            out = torch.softmax(self.forward(v), dim=-1)
            idx = torch.argmax(out).item() if greedy else torch.multinomial(out, 1).item()
            res.append(self.alphabet[idx])
        return "".join(res)
