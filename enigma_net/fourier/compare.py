import sys
import os
import glob
import itertools
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.alphabet3 import alphabet3
from enigma_net.fourier.net import EnigmaNet

def compare(weights_path=None, config=alphabet3):
    if not weights_path:
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        fourier_paths = glob.glob(os.path.join(models_dir, "fourier_learner_*.pth"))
        weights_path = max(fourier_paths) if fourier_paths else max(glob.glob(os.path.join(models_dir, "learner_*.pth")))
        
    print(f"Loading weights from {weights_path}")
    state_dict = torch.load(weights_path)
    trainable_reflector = any("reflector_layer.logits" in k for k in state_dict.keys())
    learner = EnigmaNet(config, trainable_reflector=trainable_reflector)
    learner.load_state_dict(state_dict)
    learner.eval()
    
    target = config.build()
    target_reflector = torch.from_numpy(target.reflector.matrix).float()
    target_plugboard = torch.from_numpy(target.plugboard.matrix).float()
    target_wiring = [torch.from_numpy(r.matrix).float() for r in target.rotors]
    
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
        
        E_learner = learner.forward_matrix(pos)
        M_fwd = torch.eye(len(config.alphabet))
        for W, r in zip(target_wiring, target.rotors):
            p = int(r.position)
            M_fwd = M_fwd @ torch.roll(W, shifts=(-p, -p), dims=(0, 1))
        E_target = M_fwd.T @ target_reflector @ M_fwd
        if target_plugboard is not None:
            E_target = target_plugboard @ E_target @ target_plugboard
            
        mismatches += torch.sum(torch.argmax(E_learner, dim=0) != torch.argmax(E_target, dim=0)).item()
        frob_diff += torch.norm(E_learner - E_target).item()
            
    print("argmax models are identical." if mismatches == 0 else f"Failure: Found {mismatches} mismatches.")
    print(f"Frobenius norm diff: {frob_diff:.4f}")

if __name__ == "__main__":
    compare()
