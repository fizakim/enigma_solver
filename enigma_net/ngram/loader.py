import math
import torch

def load_ngram_logprobs(path, alphabet_size=26, device="cpu"):
    log10_probs = torch.load(path, map_location="cpu")
    return (log10_probs.float() * math.log(10.0)).contiguous().to(device)

