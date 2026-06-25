import math
import torch
import torch.nn as nn
from .mappings import get_mapping

def get_logits(n, perm):
    spectrum = torch.fft.rfft(perm)
    spectral = spectrum[1:]
    return torch.cat([spectral.real, spectral.imag if n % 2 != 0 else spectral.imag[:-1]])

class Reflector(nn.Module):
    def __init__(self, size, target_reflector=None, tau=0.1, mapping_type="softmax"):
        super().__init__()
        self.n = size
        self.tau = tau
        self.dc = size * (size - 1) / 2.0
        self.mapping = get_mapping(mapping_type, size)

        if target_reflector is not None:
            perm = target_reflector.argmax(dim=0).float()
            self.register_buffer('logits', get_logits(size, perm))
        else:
            perm = torch.arange(size).float()
            shuffled = torch.randperm(size)
            for i in range(0, size - 1, 2):
                u, v = shuffled[i].item(), shuffled[i+1].item()
                perm[u], perm[v] = float(v), float(u)
            self.logits = nn.Parameter(get_logits(size, perm))

    def get_matrix(self):
        half = self.n // 2
        real = self.logits[:half]
        imag = self.logits[half:]
        if self.n % 2 == 0:
            imag = torch.cat([imag, torch.zeros(1, device=self.logits.device)])
        spectral = torch.complex(real, imag)
        dc_tensor = torch.tensor([self.dc], dtype=spectral.dtype, device=spectral.device)
        rfft_spectrum = torch.cat([dc_tensor, spectral])
        perm_values = torch.fft.irfft(rfft_spectrum, n=self.n)
        P = self.mapping(perm_values, self.tau)
        return (P + P.T) / 2.0

    def forward(self, v):
        return self.get_matrix() @ v

class Rotor(nn.Module):
    def __init__(self, size, target_wiring=None, tau=0.1, iterations=10, noise_scale=1.0, mapping_type="softmax"):
        super().__init__()
        self.n = size
        self.tau = tau
        self.dc = size * (size - 1) / 2.0
        self.mapping = get_mapping(mapping_type, size)

        if target_wiring is not None:
            perm = target_wiring.argmax(dim=0).float()
            self.register_buffer('logits', get_logits(size, perm))
        else:
            perm = torch.randperm(size).float()
            self.logits = nn.Parameter(get_logits(size, perm))

    def get_permutation_values(self, position=0):
        pos_tensor = torch.as_tensor(position, dtype=torch.float32, device=self.logits.device)
        half = self.n // 2
        real = self.logits[:half]
        imag = self.logits[half:]
        if self.n % 2 == 0:
            imag = torch.cat([imag, torch.zeros(1, device=self.logits.device)])
        spectral = torch.complex(real, imag)
        k = torch.arange(1, self.n // 2 + 1, dtype=torch.float32, device=self.logits.device)
        angles = 2.0 * math.pi * k * pos_tensor / self.n
        shifted = spectral * torch.exp(1j * angles)
        dc_val = self.dc - pos_tensor * self.n
        rfft_spectrum = torch.cat([dc_val.unsqueeze(0), shifted])
        return torch.fft.irfft(rfft_spectrum, n=self.n)

    def get_wiring(self, position=0):
        return self.mapping(self.get_permutation_values(position), self.tau)

    def forward(self, v, position):
        return self.get_wiring(position) @ v

    def backward_pass(self, v, position):
        return self.get_wiring(position).T @ v

class EnigmaNet(nn.Module):
    def __init__(self, config, load_target=False, tau=0.1, iterations=10, trainable_rotors=None, trainable_reflector=False, noise_scale=1.0, mapping_type="softmax"):
        super().__init__()
        self.config = config
        self.n = len(config.alphabet)
        self.alphabet = config.alphabet
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.tau = tau

        self.rotors = nn.ModuleList([
            Rotor(
                self.n,
                target_wiring=torch.from_numpy(config.wiring_to_matrix(r.wiring)).float() if load_target else None,
                tau=tau,
                mapping_type=mapping_type
            )
            for r in config.rotors
        ])

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
        elif trainable_rotors is not None:
            for i, r in enumerate(self.rotors):
                if i not in trainable_rotors and r.logits is not None:
                    r.logits.requires_grad = False

    @property
    def reflector(self):
        return self.reflector_layer.get_matrix()

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
        v = self.reflector_layer(v)
        for r, pos in zip(self.rotors, self.positions):
            v = r.backward_pass(v, pos)
        return v

    def forward_matrix(self, positions, wirings=None, reflector=None):
        self.reset(positions)
        self.step()
        if wirings is not None:
            W_effective = [torch.roll(w, shifts=(-pos, -pos), dims=(0, 1)) for w, pos in zip(wirings, self.positions)]
        else:
            W_effective = [r.get_wiring(pos) for r, pos in zip(self.rotors, self.positions)]

        if reflector is None:
            reflector = self.reflector

        M = torch.eye(self.n, device=reflector.device)
        for w in reversed(W_effective):
            M = w @ M
        M = reflector @ M
        for w in W_effective:
            M = w.T @ M
        return M

    def reset(self, positions=None):
        if positions is None:
            positions = [0] * len(self.rotors)
        self.positions = [self.config.parse_position(p) for p in positions]

    def set_tau(self, tau):
        self.tau = tau
        for r in self.rotors:
            r.tau = tau
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
