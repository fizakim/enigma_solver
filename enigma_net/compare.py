import sys
import os
import glob
import itertools
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config3 import config3
from enigma_net.enigma_net import EnigmaNet

def compare(weights_path=None, config=config3):
    if not weights_path:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        weights_path = max(glob.glob(os.path.join(models_dir, "learner_*.pth")))
        
    learner = EnigmaNet(config)
    learner.load_state_dict(torch.load(weights_path))
    learner.eval()
    
    target = config.build()
    
    target_reflector = torch.from_numpy(target.reflector.matrix).float()
    target_plugboard = torch.from_numpy(target.plugboard.matrix).float()
    target_wiring = [torch.from_numpy(r.matrix).float() for r in target.rotors]
    
    learner_reflector = learner.reflector
    learner_wiring = [r.get_wiring() for r in learner.rotors]
    
    def compute_matrix(wirings, reflector, positions, plugboard=None):
        M_fwd = torch.eye(reflector.shape[0])
        for W, p in zip(wirings, positions):
            M_fwd = M_fwd @ torch.roll(W, shifts=(-p, -p), dims=(0, 1))
        E = M_fwd.T @ reflector @ M_fwd
        return plugboard @ E @ plugboard if plugboard is not None else E

    mismatches = 0
    frob_diff = 0.0
    all_positions = itertools.product(range(len(config.alphabet)), repeat=len(config.rotors))
    
    for pos in all_positions:
        learner.reset(pos)
        learner.step()
        
        target.reset(pos)
        for r in reversed(target.rotors):
            if not r.step():
                break
        
        E_learner = compute_matrix(learner_wiring, learner_reflector, [int(p) for p in learner.positions])
        E_target = compute_matrix(target_wiring, target_reflector, [int(r.position) for r in target.rotors], target_plugboard)
        
        mismatches += torch.sum(torch.argmax(E_learner, dim=0) != torch.argmax(E_target, dim=0)).item()
        frob_diff += torch.norm(E_learner - E_target).item()
            
    print("argmax models are identical." if mismatches == 0 else f"Failure: Found {mismatches} mismatches.")
    print(f"Frobenius norm diff: {frob_diff:.4f}")


if __name__ == "__main__":
    compare()
