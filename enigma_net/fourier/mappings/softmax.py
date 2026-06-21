import torch
from .base import VectorMatrixMapping

class SoftmaxMapping(VectorMatrixMapping):
    def forward(self, perm_values, tau):
        targets = torch.arange(self.n, dtype=torch.float32, device=perm_values.device)
        diff = targets.unsqueeze(1) - perm_values.unsqueeze(0)
        wrapped = diff - self.n * torch.round(diff / self.n)
        scores = -wrapped.pow(2) / (2.0 * (tau ** 2))
        return torch.softmax(scores, dim=0)
