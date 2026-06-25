import torch
import torch.nn as nn

class VectorMatrixMapping(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.n = size

    def forward(self, perm_values, tau):
        raise NotImplementedError

class LinearMapping(VectorMatrixMapping):
    def forward(self, perm_values, tau):
        targets = torch.arange(self.n, dtype=torch.float32, device=perm_values.device)
        diff = targets.unsqueeze(1) - perm_values.unsqueeze(0)
        wrapped = diff - self.n * torch.round(diff / self.n)
        scores = 1.0 / (wrapped.pow(2) + tau ** 2)
        return scores / scores.sum(dim=0, keepdim=True)

class SoftmaxMapping(VectorMatrixMapping):
    def forward(self, perm_values, tau):
        targets = torch.arange(self.n, dtype=torch.float32, device=perm_values.device)
        diff = targets.unsqueeze(1) - perm_values.unsqueeze(0)
        wrapped = diff - self.n * torch.round(diff / self.n)
        scores = -wrapped.pow(2) / (2.0 * (tau ** 2))
        return torch.softmax(scores, dim=0)

def get_mapping(mapping_type, size):
    if mapping_type == "softmax":
        return SoftmaxMapping(size)
    if mapping_type == "linear":
        return LinearMapping(size)
    raise ValueError(f"Unknown mapping type: '{mapping_type}'")
