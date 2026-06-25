import itertools
import torch

from config.alphabet3 import alphabet3
from comparison.utils import find_latest_weights, compute_target_matrix

def compare(weights_path=None, config=alphabet3):
    if not weights_path:
        weights_path = find_latest_weights(["dft_learner_*.pth", "q_net_learner_*.pth"])

    state_dict = torch.load(weights_path, map_location="cpu")
    trainable_reflector = (
        any("reflector_layer.logits" in k for k in state_dict.keys())
        or any(k in ("R_real", "R_imag") for k in state_dict.keys())
    )

    is_q_net = any(k == "rotors.0.Q_real" for k in state_dict.keys())
    if is_q_net:
        from enigma_net.fourier.q_net import QNet
        learner = QNet(config, trainable_reflector=trainable_reflector)
    else:
        from enigma_net.fourier.dft_net import EnigmaNet
        learner = EnigmaNet(config, trainable_reflector=trainable_reflector)

    learner.load_state_dict(state_dict)
    learner.eval()

    target = config.build()
    target_reflector = torch.from_numpy(target.reflector.matrix).float()
    target_plugboard = torch.from_numpy(target.plugboard.matrix).float()
    target_wiring = [torch.from_numpy(r.matrix).float() for r in target.rotors]

    mismatches = 0
    frob_diff = 0.0

    for pos in itertools.product(range(len(config.alphabet)), repeat=len(config.rotors)):
        learner.reset(pos)
        learner.step()

        target.reset(pos)
        for r in reversed(target.rotors):
            if not r.step():
                break

        E_learner = learner.forward_matrix(pos)
        E_target = compute_target_matrix(target_wiring, target_reflector, [int(r.position) for r in target.rotors], target_plugboard)

        mismatches += torch.sum(torch.argmax(E_learner, dim=0) != torch.argmax(E_target, dim=0)).item()
        frob_diff += torch.norm(E_learner - E_target).item()

    print("argmax models are identical." if mismatches == 0 else f"Failure: Found {mismatches} mismatches.")
    print(f"Frobenius norm diff: {frob_diff:.4f}")
