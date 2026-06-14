import torch
from enigma_net.permutation.permutation_core import Permutation

class PermutationSampler:
    @staticmethod
    def sample(scores, temperature=1.0):
        n = scores.shape[0]
        scores = scores.clone().clamp(min=1e-12)
        if temperature != 1.0:
            scores = scores ** (1.0 / temperature)
        
        indices = []
        available = torch.ones(n, dtype=torch.bool, device=scores.device)
        for i in range(n):
            row_scores = scores[i] * available.float()
            row_sum = row_scores.sum()
            probs = row_scores / row_sum if row_sum > 0 else available.float() / available.sum()
            j = torch.multinomial(probs, 1).item()
            indices.append(j)
            available[j] = False
        return Permutation(indices)
    
    @staticmethod
    def greedy(scores):
        n = scores.shape[0]
        indices = []
        available = torch.ones(n, dtype=torch.bool, device=scores.device)
        for i in range(n):
            row_scores = scores[i].clone()
            row_scores[~available] = -float('inf')
            j = torch.argmax(row_scores).item()
            indices.append(j)
            available[j] = False
        return Permutation(indices)
    
    @staticmethod
    def sample_k(scores, k, temperature=1.0):
        return [PermutationSampler.sample(scores, temperature) for _ in range(k)]
