import math
import torch
import torch.nn as nn
from .mappings import get_mapping

def get_logits(n, perm):
    spectrum = torch.fft.rfft(perm)
    spectral = spectrum[1:]
    return torch.cat([spectral.real, spectral.imag if n % 2 != 0 else spectral.imag[:-1]])

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
        perm_values = self.get_permutation_values(position)
        return self.mapping(perm_values, self.tau)

    def forward(self, v, position):
        return self.get_wiring(position) @ v

    def backward_pass(self, v, position):
        return self.get_wiring(position).T @ v
