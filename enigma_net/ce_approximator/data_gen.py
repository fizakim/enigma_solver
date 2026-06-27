import random
from collections import Counter

import torch
import torch.nn.functional as F

from enigma_net.fourier.q_net import QNet


def corpus_unigram_prior(corpus, char_to_idx, device, eps=1e-6):
    n = len(char_to_idx)
    counts = torch.full((n,), eps)
    for ch, k in Counter(corpus).items():
        if ch in char_to_idx:
            counts[char_to_idx[ch]] += k
    return (counts / counts.sum()).to(device)


def make_random_qnet(config, device):
    return QNet(config, load_target=False).to(device)


def make_random_target(config, device):
    net = QNet(config, load_target=False).to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


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


@torch.no_grad()
def _encrypt(target, plain_list, positions):
    target.reset(positions)
    return target.encrypt_sequence(plain_list).argmax(dim=-1).tolist()


def _sample_example(config, corpus, char_to_idx, n_rotors, seq_len, device):
    n = len(config.alphabet)
    positions = [random.randint(0, n - 1) for _ in range(n_rotors)]
    start = random.randint(0, len(corpus) - seq_len - 1)
    plaintext = corpus[start: start + seq_len]
    plain_list = [char_to_idx[c] for c in plaintext]
    plain_idx = torch.tensor(plain_list, dtype=torch.long, device=device)
    target = make_random_target(config, device)
    cipher_idx = _encrypt(target, plain_list, positions)
    return cipher_idx, plain_idx, positions, target


@torch.no_grad()
def _windows(logits, plain_idx, cipher_idx, step_pos, state_vec, block_size):
    T, n = logits.shape
    n_full = T // block_size
    if n_full == 0:
        return None
    L = n_full * block_size
    R = step_pos.shape[-1]
    cipher_t = torch.as_tensor(cipher_idx, dtype=torch.long, device=logits.device)
    X = logits[:L].reshape(n_full, block_size, n).cpu().float()
    Y = plain_idx[:L].reshape(n_full, block_size).cpu().long()
    C = cipher_t[:L].reshape(n_full, block_size).cpu().long()
    P = step_pos[:L].reshape(n_full, block_size, R).cpu().long()
    S = state_vec.unsqueeze(0).expand(n_full, -1).cpu().float()
    ce = F.cross_entropy(logits[:L], plain_idx[:L], reduction="none")
    ce = ce.reshape(n_full, block_size).mean(1).cpu().float()
    return X, Y, C, P, S, ce


@torch.no_grad()
def _snapshot(candidate, cipher_idx, plain_idx, positions, block_size):
    candidate.eval()
    candidate.reset(positions)
    logits = candidate.encrypt_sequence(cipher_idx)
    step_pos = candidate.step_positions(len(cipher_idx))
    return _windows(logits, plain_idx, cipher_idx, step_pos, candidate.state_features(), block_size)


def generate_dataset(
    config, corpus, char_to_idx, device,
    block_size: int = 128,
    windows_per_candidate: int = 8,
    n_random: int = 300,
    n_traj: int = 150,
    traj_snapshots: int = 10,
    traj_opt_steps: int = 400,
    traj_lr: float = 1e-3,
    n_near: int = 300,
    n_adv: int = 300,
):
    n_rotors = len(config.rotors)
    seq_len = block_size * windows_per_candidate

    Xs, Ys, Cs, Ps, Ss, CEs = [], [], [], [], [], []

    def _add(win):
        if win is not None:
            Xs.append(win[0]); Ys.append(win[1]); Cs.append(win[2])
            Ps.append(win[3]); Ss.append(win[4]); CEs.append(win[5])

    print(f"Generating {n_random} random-wiring candidates (fresh key each)...")
    for _ in range(n_random):
        cipher_idx, plain_idx, positions, _target = _sample_example(
            config, corpus, char_to_idx, n_rotors, seq_len, device)
        _add(_snapshot(make_random_qnet(config, device), cipher_idx, plain_idx, positions, block_size))

    print(f"Generating {n_traj} deploy-matched trajectories x {traj_snapshots} snapshots "
          f"(lr={traj_lr}, {traj_opt_steps} steps)...")
    snap_at = sorted(set(
        round(i * traj_opt_steps / (traj_snapshots - 1)) for i in range(traj_snapshots)
    )) if traj_snapshots > 1 else [traj_opt_steps]
    for t in range(n_traj):
        cipher_idx, plain_idx, positions, _target = _sample_example(
            config, corpus, char_to_idx, n_rotors, seq_len, device)
        candidate = make_random_qnet(config, device)
        opt = torch.optim.Adam(candidate.parameters(), lr=traj_lr)
        for step in range(traj_opt_steps + 1):
            candidate.reset(positions)
            logits = candidate.encrypt_sequence(cipher_idx)
            if step in snap_at:
                with torch.no_grad():
                    sp = candidate.step_positions(len(cipher_idx))
                    _add(_windows(logits.detach(), plain_idx, cipher_idx, sp,
                                  candidate.state_features(), block_size))
            if step < traj_opt_steps:
                opt.zero_grad()
                F.cross_entropy(logits, plain_idx).backward()
                opt.step()
        if (t + 1) % 25 == 0:
            print(f"  trajectory {t + 1}/{n_traj}")

    print(f"Generating {n_near} near-true candidates...")
    for _ in range(n_near):
        cipher_idx, plain_idx, positions, target = _sample_example(
            config, corpus, char_to_idx, n_rotors, seq_len, device)
        net = make_near_true_qnet(config, device, random.uniform(1e-4, 0.05), target)
        _add(_snapshot(net, cipher_idx, plain_idx, positions, block_size))

    print(f"Generating {n_adv} adversarial partial-key candidates...")
    for _ in range(n_adv):
        cipher_idx, plain_idx, positions, target = _sample_example(
            config, corpus, char_to_idx, n_rotors, seq_len, device)
        net = make_partial_key_qnet(config, device, target, noise_scale=random.uniform(0.0, 0.03))
        _add(_snapshot(net, cipher_idx, plain_idx, positions, block_size))

    X = torch.cat(Xs, dim=0)
    Y = torch.cat(Ys, dim=0)
    C = torch.cat(Cs, dim=0)
    P = torch.cat(Ps, dim=0)
    S = torch.cat(Ss, dim=0)
    CE = torch.cat(CEs, dim=0)
    print(f"Dataset complete: {len(X)} windows  (block_size={block_size})  "
          f"true CE range {CE.min():.2f}..{CE.max():.2f}")
    return X, Y, C, P, S, CE


def generate_onpolicy_candidates(
    approx, config, corpus, char_to_idx, device,
    block_size: int = 128,
    windows_per_candidate: int = 8,
    n_candidates: int = 120,
    attack_steps: int = 150,
    attack_lr: float = 1e-3,
    snapshots: int = 6,
):
    n_rotors = len(config.rotors)
    seq_len = block_size * windows_per_candidate

    was_training = approx.denoiser.training
    approx.denoiser.eval()

    snap_at = sorted(set(
        round(i * attack_steps / (snapshots - 1)) for i in range(snapshots)
    )) if snapshots > 1 else [attack_steps]

    Xs, Ys, Cs, Ps, Ss, CEs = [], [], [], [], [], []

    def _add(win):
        if win is not None:
            Xs.append(win[0]); Ys.append(win[1]); Cs.append(win[2])
            Ps.append(win[3]); Ss.append(win[4]); CEs.append(win[5])

    print(f"Generating {n_candidates} on-policy candidates x {len(snap_at)} snapshots "
          f"({attack_steps} attack steps, lr={attack_lr})...")
    for c_i in range(n_candidates):
        cipher_idx, plain_idx, positions, _target = _sample_example(
            config, corpus, char_to_idx, n_rotors, seq_len, device)
        cipher_t = torch.tensor(cipher_idx, dtype=torch.long, device=device).unsqueeze(0)
        learner = make_random_qnet(config, device)
        opt = torch.optim.Adam(learner.parameters(), lr=attack_lr)
        for step in range(attack_steps + 1):
            learner.reset(positions)
            logits = learner.encrypt_sequence(cipher_idx)
            if step in snap_at:
                with torch.no_grad():
                    sp = learner.step_positions(len(cipher_idx))
                    _add(_windows(logits.detach(), plain_idx, cipher_idx, sp,
                                  learner.state_features(), block_size))
            if step < attack_steps:
                pos = learner.step_positions(len(cipher_idx)).unsqueeze(0)
                state = learner.state_features().unsqueeze(0)
                loss = approx(logits.unsqueeze(0), cipher=cipher_t,
                              positions=pos, qnet_state=state).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        if (c_i + 1) % 25 == 0:
            print(f"  candidate {c_i + 1}/{n_candidates}")

    if was_training:
        approx.denoiser.train()

    X = torch.cat(Xs, dim=0)
    Y = torch.cat(Ys, dim=0)
    C = torch.cat(Cs, dim=0)
    P = torch.cat(Ps, dim=0)
    S = torch.cat(Ss, dim=0)
    CE = torch.cat(CEs, dim=0)
    print(f"On-policy batch: {len(X)} windows  true CE range {CE.min():.2f}..{CE.max():.2f}")
    return X, Y, C, P, S, CE
