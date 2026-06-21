import os
import glob
import torch

def find_latest_weights(patterns, model_dirs=None):
    if model_dirs is None:
        # Default to root models directory and enigma_net/models directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_dirs = [
            os.path.join(root_dir, "models"),
            os.path.join(root_dir, "enigma_net", "models")
        ]
    
    all_files = []
    for d in model_dirs:
        if os.path.exists(d):
            for pattern in patterns:
                all_files.extend(glob.glob(os.path.join(d, pattern)))
                
    if not all_files:
        raise FileNotFoundError(f"No weights matching patterns {patterns} found in {model_dirs}")
    return max(all_files, key=os.path.getmtime)

def compute_target_matrix(wirings, reflector, positions, plugboard=None):
    M_fwd = torch.eye(reflector.shape[0])
    for W, p in zip(wirings, positions):
        M_fwd = M_fwd @ torch.roll(W, shifts=(-p, -p), dims=(0, 1))
    E = M_fwd.T @ reflector @ M_fwd
    if plugboard is not None:
        E = plugboard @ E @ plugboard
    return E
