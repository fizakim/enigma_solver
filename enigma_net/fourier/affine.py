"""Multiplier (affine) wiring anchors — the structured Q-basins.

A multiplier permutation ``x -> a*x (mod n)`` with ``gcd(a, n) == 1`` is an exact
permutation matrix whose Fourier transform is itself a frequency permutation,
``Q_a[k, l] = delta[l == a*k (mod n)]``. The ``phi(n)`` multipliers are the
restricted "Q-basins" used to initialise the per-position wiring search; for n=26
the group ``Z_n*`` is cyclic, so they are exactly the powers of a primitive root,
``M_a = exp(j * L_mult)`` — the multiplicative analog of the integer-``phi`` rotor
positions.
"""

from math import gcd

import torch

from .q_net.net import _make_dft

__all__ = [
    "multiplier_units",
    "multiplier_perm_matrix",
    "multiplier_anchor_Q",
    "affine_anchor_Q",
    "affine_wiring_string",
    "make_dft",
]

# Re-export so callers don't reach into q_net for the DFT convention.
make_dft = _make_dft


def multiplier_units(n):
    """The ``phi(n)`` multipliers ``a`` with ``gcd(a, n) == 1`` — the Q-basins."""
    return [a for a in range(1, n) if gcd(a, n) == 1]


def multiplier_perm_matrix(a, n):
    """Spatial permutation matrix for ``x -> a*x (mod n)``. Returns ``[n, n]`` float."""
    P = torch.zeros(n, n)
    for col in range(n):
        P[(a * col) % n, col] = 1.0
    return P


def multiplier_anchor_Q(a, F, F_inv):
    """Exact Fourier anchor ``Q_a = F P_a F^{-1}`` (complex). ``Q_a[k, l] = delta[l == a*k]``."""
    n = F.shape[0]
    return F @ multiplier_perm_matrix(a, n).to(F.dtype) @ F_inv


def affine_anchor_Q(a, b, F, F_inv):
    """Fourier anchor for affine wiring x → ax+b (mod n).

    Q_{a,b}[k,l] = exp(−2πi·k·b/n) · Q_a[k,l]  (phase-scales each row of Q_a).
    """
    import math
    Q_mult = multiplier_anchor_Q(a, F, F_inv)
    n = F.shape[0]
    k = torch.arange(n, dtype=F.real.dtype, device=F.device)
    phase = torch.exp(-2j * math.pi * k * b / n)   # shape [n], complex
    return phase.unsqueeze(1) * Q_mult              # row-wise multiply → [n, n]


def affine_wiring_string(a, b, alphabet):
    """Wiring string for ``x -> a*x + b (mod n)``; plugs into ``RotorConfig(wiring=...)``."""
    n = len(alphabet)
    return "".join(alphabet[(a * j + b) % n] for j in range(n))
