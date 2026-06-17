import math
import torch
import torch.nn as nn

def get_logits(n, perm):
    spectrum = torch.fft.rfft(perm)
    spectral = spectrum[1:]
    return torch.cat([spectral.real, spectral.imag if n % 2 != 0 else spectral.imag[:-1]])

class Reflector(nn.Module):
    def __init__(self, size, target_reflector=None, tau=0.1):
        super().__init__()
        self.n = size
        self.tau = tau
        self.dc = size * (size - 1) / 2.0

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
        
        targets = torch.arange(self.n, dtype=torch.float32, device=perm_values.device)
        diff = targets.unsqueeze(1) - perm_values.unsqueeze(0)
        scores = torch.cos(2.0 * math.pi * diff / self.n) / self.tau
        P = torch.softmax(scores, dim=0)
        return (P + P.T) / 2.0

    def forward(self, v):
        return self.get_matrix() @ v
