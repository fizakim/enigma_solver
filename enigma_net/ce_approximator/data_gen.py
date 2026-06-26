import random

import torch
import torch.nn.functional as F

from enigma_net.fourier.q_net import QNet


def make_random_qnet(config, device):
    return QNet(config, load_target=False).to(device)


def make_near_true_qnet(config, device, noise_scale, true_net):
    net = QNet(config, load_target=False).to(device)
    with torch.no_grad():
        for i, rotor in enumerate(net.rotors):
            Q_true_r = true_net.rotors[i].Q_real
            Q_true_i = true_net.rotors[i].Q_imag
            rotor.Q_real.copy_(Q_true_r + noise_scale * torch.randn_like(Q_true_r))
            rotor.Q_imag.copy_(Q_true_i + noise_scale * torch.randn_like(Q_true_i))
    return net


def make_partial_key_qnet(config, device, true_net, noise_scale=0.0):
    net = QNet(config, load_target=False).to(device)
    n_rotors = len(net.rotors)
    n_correct = random.randint(1, max(1, n_rotors - 1))
    correct = set(random.sample(range(n_rotors), n_correct))
    with torch.no_grad():
        for i, rotor in enumerate(net.rotors):
            if i in correct:
                r = true_net.rotors[i].Q_real
                im = true_net.rotors[i].Q_imag
                rotor.Q_real.copy_(r + noise_scale * torch.randn_like(r))
                rotor.Q_imag.copy_(im + noise_scale * torch.randn_like(im))
    return net


@torch.no_grad()
def perm_hardness(net) -> float:
    scores = []
    for rotor in net.rotors:
        M = rotor.get_spatial_matrix().abs()
        scores.append(M.max(dim=0).values.mean())
    return float(1.0 - torch.stack(scores).mean().clamp(0.0, 1.0))


def _sample_pair(config, corpus, char_to_idx, n_rotors, seq_len, device):
    positions = [random.randint(0, len(config.alphabet) - 1) for _ in range(n_rotors)]
    start = random.randint(0, len(corpus) - seq_len - 1)
    plaintext = corpus[start: start + seq_len]
    target = config.build(positions)
    target.reset(positions)
    ciphertext = target.encrypt(plaintext)
    cipher_idx = [char_to_idx[c] for c in ciphertext]
    plain_idx = torch.tensor([char_to_idx[c] for c in plaintext], dtype=torch.long, device=device)
    return cipher_idx, plain_idx, positions


@torch.no_grad()
def _windows(logits, plain_idx, block_size):
    T, n = logits.shape
    n_full = T // block_size
    if n_full == 0:
        return None
    L = n_full * block_size
    X = logits[:L].reshape(n_full, block_size, n).cpu().float()
    Y = plain_idx[:L].reshape(n_full, block_size).cpu().long()
    ce = F.cross_entropy(logits[:L], plain_idx[:L], reduction="none")
    ce = ce.reshape(n_full, block_size).mean(1).cpu().float()
    return X, Y, ce


@torch.no_grad()
def _snapshot(candidate, cipher_idx, plain_idx, positions, block_size):
    candidate.eval()
    candidate.reset(positions)
    logits = candidate.encrypt_sequence(cipher_idx)
    return _windows(logits, plain_idx, block_size)


def generate_dataset(
    config, corpus, char_to_idx, device,
    block_size: int = 128,
    windows_per_candidate: int = 8,
    n_random: int = 300,
    n_traj: int = 150,
    traj_snapshots: int = 10,
    traj_opt_steps: int = 50,
    traj_lr: float = 0.02,
    n_near: int = 300,
    n_adv: int = 300,
):
    n_rotors = len(config.rotors)
    seq_len = block_size * windows_per_candidate

    true_net = QNet(config, load_target=True).to(device)
    true_net.eval()

    Xs, Ys, CEs = [], [], []

    def _add(win):
        if win is not None:
            Xs.append(win[0]); Ys.append(win[1]); CEs.append(win[2])

    print(f"Generating {n_random} random-wiring candidates...")
    for _ in range(n_random):
        cipher_idx, plain_idx, positions = _sample_pair(config, corpus, char_to_idx, n_rotors, seq_len, device)
        _add(_snapshot(make_random_qnet(config, device), cipher_idx, plain_idx, positions, block_size))

    print(f"Generating {n_traj} true-CE trajectories x {traj_snapshots} snapshots...")
    snap_at = sorted(set(
        round(i * traj_opt_steps / (traj_snapshots - 1)) for i in range(traj_snapshots)
    )) if traj_snapshots > 1 else [traj_opt_steps]
    for t in range(n_traj):
        cipher_idx, plain_idx, positions = _sample_pair(config, corpus, char_to_idx, n_rotors, seq_len, device)
        candidate = make_random_qnet(config, device)
        candidate.reset(positions)
        opt = torch.optim.Adam(candidate.parameters(), lr=traj_lr)
        for step in range(traj_opt_steps + 1):
            candidate.reset(positions)
            logits = candidate.encrypt_sequence(cipher_idx)
            if step in snap_at:
                _add(_windows(logits.detach(), plain_idx, block_size))
            if step < traj_opt_steps:
                opt.zero_grad()
                F.cross_entropy(logits, plain_idx).backward()
                opt.step()
        if (t + 1) % 25 == 0:
            print(f"  trajectory {t + 1}/{n_traj}")

    print(f"Generating {n_near} near-true candidates...")
    for _ in range(n_near):
        cipher_idx, plain_idx, positions = _sample_pair(config, corpus, char_to_idx, n_rotors, seq_len, device)
        net = make_near_true_qnet(config, device, random.uniform(1e-4, 0.05), true_net)
        _add(_snapshot(net, cipher_idx, plain_idx, positions, block_size))

    print(f"Generating {n_adv} adversarial partial-key candidates...")
    for _ in range(n_adv):
        cipher_idx, plain_idx, positions = _sample_pair(config, corpus, char_to_idx, n_rotors, seq_len, device)
        net = make_partial_key_qnet(config, device, true_net, noise_scale=random.uniform(0.0, 0.03))
        _add(_snapshot(net, cipher_idx, plain_idx, positions, block_size))

    X = torch.cat(Xs, dim=0)
    Y = torch.cat(Ys, dim=0)
    CE = torch.cat(CEs, dim=0)
    print(f"Dataset complete: {len(X)} windows  (block_size={block_size})  "
          f"true CE range {CE.min():.2f}..{CE.max():.2f}")
    return X, Y, CE
