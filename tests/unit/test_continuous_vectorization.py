import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import numpy as np
import random
from config.alphabet3 import alphabet3
from config.alphabet5 import alphabet5
from enigma_net.fourier.continuous.net import ContinuousQNet

# We implement the exact old implementation of forward and encrypt_sequence
# to compare against our new vectorized & stabilized version.
def old_forward(net, v, step_offset):
    # v: shape (n,)
    # step_offset: shape (C, num_rotors)
    u = net.F @ v.to(net.F.dtype)
    u = u.unsqueeze(0).expand(net.phi.shape[0], -1) # shape (C, n)
    
    for i in range(net.num_rotors - 1, -1, -1):
        Q = net.rotors[i].get_Q()
        if Q.ndim == 2:
            Q = Q.unsqueeze(0)
        
        phi_eff = net.phi[:, i] + step_offset[:, i].float()
        phase = net.omega ** phi_eff.unsqueeze(-1)
        u = phase * u
        u = torch.bmm(u.unsqueeze(1), Q.transpose(-2, -1)).squeeze(1)
        u = phase.conj() * u

    u = u @ net.reflector_fourier.T

    for i in range(net.num_rotors):
        Q = net.rotors[i].get_Q()
        if Q.ndim == 2:
            Q = Q.unsqueeze(0)
            
        phi_eff = net.phi[:, i] + step_offset[:, i].float()
        phase = net.omega ** phi_eff.unsqueeze(-1)
        u = phase * u
        u = torch.bmm(u.unsqueeze(1), Q.conj()).squeeze(1)
        u = phase.conj() * u

    return (u @ net.F_inv.T).real.float()

def old_encrypt_sequence(net, input_indices):
    T = len(input_indices)
    step_offsets = net.precompute_steps(T)
    outputs = []
    for t in range(T):
        v = torch.zeros(net.n, device=net.phi.device)
        v[input_indices[t]] = 1.0
        outputs.append(old_forward(net, v, step_offsets[t]))
    return torch.stack(outputs)

def test_vectorization_and_phase_equivalence():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    config = alphabet5
    n = len(config.alphabet)
    num_rotors = len(config.rotors)
    
    # Generate random starting hypotheses (basins)
    all_positions = []
    for _ in range(10):
        all_positions.append([random.randint(0, n - 1) for _ in range(num_rotors)])
        
    net = ContinuousQNet(config, initial_positions=all_positions)
    
    # Random input sequence
    T = 15
    input_indices = [random.randint(0, n - 1) for _ in range(T)]
    
    # Run old (loop-based, power phase) implementation
    old_logits = old_encrypt_sequence(net, input_indices)
    
    # Run new (vectorized, exp phase) implementation
    new_logits = net.encrypt_sequence(input_indices)
    
    # Verify shapes match
    assert old_logits.shape == new_logits.shape, f"Shape mismatch: {old_logits.shape} vs {new_logits.shape}"
    
    # Verify values are extremely close (allowing for tiny numeric differences between pow and exp)
    max_diff = torch.max(torch.abs(old_logits - new_logits)).item()
    print(f"Max absolute difference: {max_diff:.2e}")
    assert torch.allclose(old_logits, new_logits, atol=1e-5), f"Logits mismatch, max diff: {max_diff}"

def test_encrypt_sequence_slice_equivalence():
    torch.manual_seed(99)
    np.random.seed(99)
    random.seed(99)

    config = alphabet5
    n = len(config.alphabet)
    num_rotors = len(config.rotors)
    all_positions = [[random.randint(0, n - 1) for _ in range(num_rotors)] for _ in range(20)]
    net = ContinuousQNet(config, initial_positions=all_positions)

    T = 12
    input_indices = [random.randint(0, n - 1) for _ in range(T)]
    c_indices = torch.tensor([0, 3, 7, 15, 19])

    full_out = net.encrypt_sequence(input_indices)                          # [T, 20, n]
    slice_out = net.encrypt_sequence_slice(input_indices, c_indices)        # [T, 5, n]

    assert full_out.shape == (T, 20, n)
    assert slice_out.shape == (T, 5, n)
    max_diff = torch.max(torch.abs(full_out[:, c_indices, :] - slice_out)).item()
    print(f"Slice max absolute difference: {max_diff:.2e}")
    assert torch.allclose(full_out[:, c_indices, :], slice_out, atol=1e-5), \
        f"Slice mismatch, max diff: {max_diff}"


if __name__ == "__main__":
    test_vectorization_and_phase_equivalence()
    print("Vectorization test passed successfully!")
    test_encrypt_sequence_slice_equivalence()
    print("Slice equivalence test passed successfully!")
