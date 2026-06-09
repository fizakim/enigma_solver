import torch
import torch.nn as nn

class Sinkhorn(nn.Module):
    def __init__(self, tau=0.1, iterations=10):
        super().__init__()
        self.tau = tau
        self.iterations = iterations

    def forward(self, x):
        log_P = x / self.tau
        for _ in range(self.iterations):
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
        return torch.exp(log_P)
