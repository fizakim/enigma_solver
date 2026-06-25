from math import gcd, pi
import torch

def multiplier_units(n):
    return [a for a in range(1, n) if gcd(a, n) == 1]

def multiplier_perm_matrix(a, n):
    P = torch.zeros(n, n)
    for col in range(n):
        P[(a * col) % n, col] = 1.0
    return P

def multiplier_anchor_Q(a, F, F_inv):
    n = F.shape[0]
    return F @ multiplier_perm_matrix(a, n).to(F.dtype) @ F_inv

def affine_anchor_Q(a, b, F, F_inv):
    Q_mult = multiplier_anchor_Q(a, F, F_inv)
    n = F.shape[0]
    k = torch.arange(n, dtype=F.real.dtype, device=F.device)
    phase = torch.exp(-2j * pi * k * b / n)
    return phase.unsqueeze(1) * Q_mult

def affine_wiring_string(a, b, alphabet):
    n = len(alphabet)
    return "".join(alphabet[(a * j + b) % n] for j in range(n))
