import torch

class Permutation:
    def __init__(self, indices):
        if isinstance(indices, torch.Tensor):
            self.indices = indices.long().detach()
        else:
            self.indices = torch.tensor(indices, dtype=torch.long)
        self.n = len(self.indices)

    def __repr__(self):
        return f"Permutation({self.indices.tolist()})"
