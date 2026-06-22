import sys
import os
import itertools
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.alphabet3 import alphabet3
from comparison.utils import find_latest_weights, compute_target_matrix


def _step_positions(positions, notches, n):
    """Apply one Enigma step (rightmost always advances, cascade on notch)."""
    pos = list(positions)
    for i in range(len(pos) - 1, -1, -1):
        at_notch = (pos[i] == notches[i])
        pos[i] = (pos[i] + 1) % n
        if not at_notch:
            break
    return pos


def _encryption_matrix(net, c, stepped_positions):
    """Build the full encryption matrix for candidate c at already-stepped integer positions."""
    n = net.n
    M_fwd = torch.eye(n)
    for i in range(net.num_rotors - 1, -1, -1):
        Q = torch.complex(net.rotors[i].Q_real[c], net.rotors[i].Q_imag[c])
        spatial = (net.F_inv @ Q @ net.F).real
        W = torch.roll(spatial, shifts=(-stepped_positions[i], -stepped_positions[i]), dims=(0, 1))
        M_fwd = M_fwd @ W
    E = M_fwd.T @ net.reflector @ M_fwd
    return E


def compare(weights_path=None, config=alphabet3):
    if not weights_path:
        patterns = ["continuous_qnet_*.pth", "continuous_learner_*.pth"]
        try:
            weights_path = find_latest_weights(patterns)
        except FileNotFoundError:
            weights_path = find_latest_weights(["learner_*.pth"])

    print(f"Loading weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")

    num_candidates = state_dict["initial_positions"].shape[0]
    num_rotors = state_dict["initial_positions"].shape[1]
    print(f"Model: {num_candidates} candidate(s), {num_rotors} rotor(s)")

    from enigma_net.fourier.continuous.net import ContinuousQNet
    dummy_positions = [[0] * num_rotors for _ in range(num_candidates)]
    learner = ContinuousQNet(config, initial_positions=dummy_positions)
    learner.load_state_dict(state_dict)
    learner.eval()

    int_positions = learner.initial_positions  # [C, num_rotors]
    n = learner.n

    print("\nFixed initial positions:")
    for c in range(num_candidates):
        pos = int_positions[c].tolist()
        print(f"  Candidate {c:>3d}: positions={pos}")

    # With phi removed there is no continuous drift; pick the single candidate (post-prune).
    best_c = 0
    best_pos = int_positions[best_c].tolist()
    print(f"\nUsing candidate {best_c} for wiring comparison")
    print(f"  Its integer positions: {best_pos}")

    target = config.build()
    target_reflector = torch.from_numpy(target.reflector.matrix).float()
    target_plugboard = torch.from_numpy(target.plugboard.matrix).float()
    target_wiring = [torch.from_numpy(r.matrix).float() for r in target.rotors]
    notches = learner.notches

    mismatches = 0
    frob_diff = 0.0
    total_positions = 0

    for pos in itertools.product(range(n), repeat=num_rotors):
        stepped = _step_positions(list(pos), notches, n)

        target.reset(pos)
        for r in reversed(target.rotors):
            if not r.step():
                break
        target_stepped = [int(r.position) for r in target.rotors]

        E_learner = _encryption_matrix(learner, best_c, stepped)
        E_target = compute_target_matrix(
            target_wiring, target_reflector, target_stepped, target_plugboard
        )

        mismatches += torch.sum(
            torch.argmax(E_learner, dim=0) != torch.argmax(E_target, dim=0)
        ).item()
        frob_diff += torch.norm(E_learner - E_target).item()
        total_positions += 1

    print(f"\nComparison over {total_positions} position combinations:")
    if mismatches == 0:
        print("  argmax models are identical.")
    else:
        print(f"  Failure: {mismatches} argmax mismatches.")
    print(f"  Total Frobenius norm diff: {frob_diff:.4f}")
    print(f"  Mean Frobenius norm diff:  {frob_diff / total_positions:.4f}")


if __name__ == "__main__":
    compare()
