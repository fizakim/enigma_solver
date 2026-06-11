import os
import sys
import glob
import itertools
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config3 import config3
from enigma_net.enigma_net import EnigmaNet

def compare(weights_path=None):
    if weights_path is None:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        weights_path = max(glob.glob(os.path.join(models_dir, "learner_*.pth")))
        
    learner = EnigmaNet(config3, load_target=False)
    learner.load_state_dict(torch.load(weights_path))
    learner.eval()
    
    target = config3.build()
    
    alphabet = config3.alphabet
    n_rotors = len(config3.rotors)
    all_positions = list(itertools.product(range(len(alphabet)), repeat=n_rotors))
    
    total_checks = 0
    mismatches = 0
    
    for pos in all_positions:
        for char in alphabet:
            learner.reset(list(pos))
            target.reset(list(pos))
            if learner.encrypt_string(char, greedy=True) != target.encrypt(char):
                print(f"Mismatch at pos={list(pos)}, input='{char}'")
                mismatches += 1
            total_checks += 1
            
    if mismatches == 0:
        print(f"Models are identical.")
    else:
        print(f"Failure: Found {mismatches} mismatches.")

if __name__ == "__main__":
    compare()
