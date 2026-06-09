import torch
import torch.nn as nn

class Sinkhorn(nn.Module):
    def __init__(self, tau: float = 0.1, iterations: int = 10):
        super().__init__()
        self.tau = tau
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_P = x / self.tau
        for i in range(self.iterations):
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
        return torch.exp(log_P)
