import torch.nn as nn

class VectorMatrixMapping(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.n = size

    def forward(self, perm_values, tau):
        raise NotImplementedError
