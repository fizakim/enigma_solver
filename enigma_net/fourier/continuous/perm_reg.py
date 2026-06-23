import torch


def permutation_regularizer(net):
    """Unsupervised per-candidate regularizer: penalizes Q matrices whose spatial
    form deviates from a valid permutation matrix.

    For a true permutation P:
      - Every row sums to 1    -> (row_sum - 1)^2 = 0
      - Every column sums to 1 -> (col_sum - 1)^2 = 0
      - Every entry is 0 or 1  -> P*(1-P) = 0

    Returns [C] float32 loss values (one per candidate). Caller should mask by
    active_mask before calling .backward() to keep frozen candidates unchanged.
    """
    device = net.rotors[0].Q_real.device
    loss = torch.zeros(net.num_candidates, device=device)
    for rotor in net.rotors:
        P = rotor.get_spatial_matrix()           # [C, n, n]
        row_dev = (P.sum(dim=-1) - 1.0) ** 2    # [C, n]
        col_dev = (P.sum(dim=-2) - 1.0) ** 2    # [C, n]
        binary  = (P * (1.0 - P)).clamp(min=0)  # [C, n, n]; 0 at 0 or 1, peak 0.25 at 0.5
        loss = loss + row_dev.mean(-1) + col_dev.mean(-1) + binary.mean((-1, -2))
    return loss
