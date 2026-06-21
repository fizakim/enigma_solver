import torch
from .base import VectorMatrixMapping

class LinearMapping(VectorMatrixMapping):
    def forward(self, perm_values, tau):
        targets = torch.arange(self.n, dtype=torch.float32, device=perm_values.device)
        diff = targets.unsqueeze(1) - perm_values.unsqueeze(0)
        wrapped = diff - self.n * torch.round(diff / self.n)
        scores = 1.0 / (wrapped.pow(2) + tau ** 2)
        return scores / scores.sum(dim=0, keepdim=True)
