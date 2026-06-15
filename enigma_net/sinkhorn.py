import torch
import torch.nn as nn

class Sinkhorn(nn.Module):
    def __init__(self, tau=0.1, iterations=5, noise_scale=1.0):
        super().__init__()
        self.tau = tau
        self.iterations = iterations
        self.noise_scale = noise_scale

    def forward(self, x):
        if self.training and self.noise_scale > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(x)))
            x = x + self.noise_scale * gumbel
        log_P = x / self.tau
        for _ in range(self.iterations):
            log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
            log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
        return torch.exp(log_P)
