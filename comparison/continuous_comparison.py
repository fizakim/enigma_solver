import itertools
import torch

from config.alphabet3 import alphabet3
from comparison.utils import find_latest_weights, compute_target_matrix

def _step_positions(positions, notches, n):
    pos = list(positions)
    for i in range(len(pos) - 1, -1, -1):
        at_notch = (pos[i] == notches[i])
        pos[i] = (pos[i] + 1) % n
        if not at_notch:
            break
    return pos

def _encryption_matrix(net, c, stepped_positions):
    n = net.n
    M_fwd = torch.eye(n)
    for i in range(net.num_rotors - 1, -1, -1):
        Q = torch.complex(net.rotors[i].Q_real[c], net.rotors[i].Q_imag[c])
        spatial = (net.F_inv @ Q @ net.F).real
        W = torch.roll(spatial, shifts=(-stepped_positions[i], -stepped_positions[i]), dims=(0, 1))
        M_fwd = M_fwd @ W
    return M_fwd.T @ net.reflector @ M_fwd

def compare(weights_path=None, config=alphabet3):
    if not weights_path:
        weights_path = find_latest_weights(["continuous_qnet_*.pth"])

    state_dict = torch.load(weights_path, map_location="cpu")
    num_candidates = state_dict["initial_positions"].shape[0]
    num_rotors = state_dict["initial_positions"].shape[1]

    from enigma_net.fourier.continuous_q_net import ContinuousQNet
    dummy_positions = [[0] * num_rotors for _ in range(num_candidates)]
    learner = ContinuousQNet(config, initial_positions=dummy_positions)
    learner.load_state_dict(state_dict)
    learner.eval()

    n = learner.n
    notches = learner.notches

    target = config.build()
    target_reflector = torch.from_numpy(target.reflector.matrix).float()
    target_plugboard = torch.from_numpy(target.plugboard.matrix).float()
    target_wiring = [torch.from_numpy(r.matrix).float() for r in target.rotors]

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

        E_learner = _encryption_matrix(learner, 0, stepped)
        E_target = compute_target_matrix(target_wiring, target_reflector, target_stepped, target_plugboard)

        mismatches += torch.sum(torch.argmax(E_learner, dim=0) != torch.argmax(E_target, dim=0)).item()
        frob_diff += torch.norm(E_learner - E_target).item()
        total_positions += 1

    print(f"Comparison over {total_positions} position combinations:")
    print("  argmax models are identical." if mismatches == 0 else f"  Failure: {mismatches} argmax mismatches.")
    print(f"  Mean Frobenius norm diff: {frob_diff / total_positions:.4f}")
