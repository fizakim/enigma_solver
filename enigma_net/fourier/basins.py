import itertools
import torch
from .continuous_q_net import ContinuousQNet
from .affine import multiplier_units, multiplier_anchor_Q, affine_anchor_Q

def basin_instances(config, positions):
    n = len(config.alphabet)
    r = len(config.rotors)
    combos = list(itertools.product(multiplier_units(n), repeat=r))
    net = ContinuousQNet(config, initial_positions=[list(positions)] * len(combos))
    F, F_inv = net.F, net.F_inv
    with torch.no_grad():
        for c, mult in enumerate(combos):
            for i, a in enumerate(mult):
                Qa = multiplier_anchor_Q(a, F, F_inv)
                net.rotors[i].Q_real.data[c] = Qa.real
                net.rotors[i].Q_imag.data[c] = Qa.imag
    net.basin_combos = combos
    return net

def affine_basin_instances(config):
    n = len(config.alphabet)
    r = len(config.rotors)
    per_rotor = list(itertools.product(multiplier_units(n), range(n)))
    combos = list(itertools.product(per_rotor, repeat=r))
    
    net = ContinuousQNet(config, initial_positions=[[0] * r for _ in combos])
    F, F_inv = net.F, net.F_inv
    with torch.no_grad():
        for c, combo in enumerate(combos):
            for i, (a, b) in enumerate(combo):
                Qab = affine_anchor_Q(a, b, F, F_inv)
                net.rotors[i].Q_real.data[c] = Qab.real
                net.rotors[i].Q_imag.data[c] = Qab.imag
    net.basin_combos = combos
    return net

def single_rotor_affine_score(config, rotor_idx, fixed_wirings, ciphertext, loss_fn, device=None):
    if device is None:
        device = torch.device("cpu")
    n = len(config.alphabet)
    r = len(config.rotors)
    per_rotor = list(itertools.product(multiplier_units(n), range(n)))
    P = len(per_rotor)

    _tmp = ContinuousQNet(config, initial_positions=[[0] * r])
    F, F_inv = _tmp.F, _tmp.F_inv

    Q_real_var = torch.empty(P, n, n)
    Q_imag_var = torch.empty(P, n, n)
    for c, (a, b) in enumerate(per_rotor):
        Qab = affine_anchor_Q(a, b, F, F_inv)
        Q_real_var[c] = Qab.real
        Q_imag_var[c] = Qab.imag
    Q_real_var = Q_real_var.to(device)
    Q_imag_var = Q_imag_var.to(device)

    net = ContinuousQNet(config, initial_positions=[[0] * r] * P).to(device)
    with torch.no_grad():
        for j in range(r):
            if j == rotor_idx:
                continue
            a_j, b_j = fixed_wirings[j]
            Qj = affine_anchor_Q(a_j, b_j, F, F_inv).to(device)
            net.rotors[j].Q_real.data[:] = Qj.real
            net.rotors[j].Q_imag.data[:] = Qj.imag

        net.rotors[rotor_idx].Q_real.data[:] = Q_real_var
        net.rotors[rotor_idx].Q_imag.data[:] = Q_imag_var

        c_idx = torch.arange(P, device=device)
        logits = net.encrypt_sequence_slice(ciphertext, c_idx).transpose(0, 1)
        losses = loss_fn(logits).cpu()

    return per_rotor, losses

def stream_affine_eval(config, ciphertext, loss_fn, batch_c=512, device=None):
    if device is None:
        device = torch.device("cpu")
    n = len(config.alphabet)
    r = len(config.rotors)
    units = multiplier_units(n)
    per_rotor = list(itertools.product(units, range(n)))
    P = len(per_rotor)

    _dummy = ContinuousQNet(config, initial_positions=[[0] * r])
    F, F_inv = _dummy.F, _dummy.F_inv

    Q_real_tbl = torch.empty(P, n, n)
    Q_imag_tbl = torch.empty(P, n, n)
    for idx, (a, b) in enumerate(per_rotor):
        Qab = affine_anchor_Q(a, b, F, F_inv)
        Q_real_tbl[idx] = Qab.real
        Q_imag_tbl[idx] = Qab.imag
    Q_real_tbl = Q_real_tbl.to(device)
    Q_imag_tbl = Q_imag_tbl.to(device)

    net = ContinuousQNet(config, initial_positions=[[0] * r] * batch_c).to(device)
    C_total = P ** r
    all_losses = torch.empty(C_total)
    ri = torch.empty(batch_c, r, dtype=torch.long, device=device)
    _c_idx = torch.arange(batch_c, device=device)

    def _flush(count):
        with torch.no_grad():
            for rot_i in range(r):
                net.rotors[rot_i].Q_real.data[:count] = Q_real_tbl[ri[:count, rot_i]]
                net.rotors[rot_i].Q_imag.data[:count] = Q_imag_tbl[ri[:count, rot_i]]
            logits = net.encrypt_sequence_slice(ciphertext, _c_idx[:count]).transpose(0, 1)
            return loss_fn(logits).cpu()

    for batch_start in range(0, C_total, batch_c):
        count = min(batch_c, C_total - batch_start)
        temp = torch.arange(batch_start, batch_start + count, dtype=torch.long, device=device)
        for rot_i in range(r - 1, -1, -1):
            ri[:count, rot_i] = temp % P
            temp = temp // P
        all_losses[batch_start:batch_start + count] = _flush(count)

    return per_rotor, all_losses

def stream_affine_combo(flat_idx, per_rotor, r):
    P = len(per_rotor)
    combo = []
    for _ in range(r):
        combo.append(per_rotor[flat_idx % P])
        flat_idx //= P
    return tuple(reversed(combo))

def eval_affine_combos(config, combos, ciphertext, loss_fn, batch_c=512, device=None):
    if device is None:
        device = torch.device("cpu")
    n = len(config.alphabet)
    r = len(config.rotors)

    per_rotor = list(itertools.product(multiplier_units(n), range(n)))
    ab_to_idx = {ab: i for i, ab in enumerate(per_rotor)}

    _dummy = ContinuousQNet(config, initial_positions=[[0] * r])
    F, F_inv = _dummy.F, _dummy.F_inv

    Q_real_tbl = torch.empty(len(per_rotor), n, n)
    Q_imag_tbl = torch.empty(len(per_rotor), n, n)
    for idx, (a, b) in enumerate(per_rotor):
        Qab = affine_anchor_Q(a, b, F, F_inv)
        Q_real_tbl[idx] = Qab.real
        Q_imag_tbl[idx] = Qab.imag
    Q_real_tbl = Q_real_tbl.to(device)
    Q_imag_tbl = Q_imag_tbl.to(device)

    C = len(combos)
    all_losses = torch.empty(C)
    net = ContinuousQNet(config, initial_positions=[[0] * r] * batch_c).to(device)

    for batch_start in range(0, C, batch_c):
        count = min(batch_c, C - batch_start)
        batch = combos[batch_start:batch_start + count]
        with torch.no_grad():
            for rot_i in range(r):
                idxs = torch.tensor(
                    [ab_to_idx[combo[rot_i]] for combo in batch],
                    dtype=torch.long, device=device,
                )
                net.rotors[rot_i].Q_real.data[:count] = Q_real_tbl[idxs]
                net.rotors[rot_i].Q_imag.data[:count] = Q_imag_tbl[idxs]
            c_idx = torch.arange(count, device=device)
            logits = net.encrypt_sequence_slice(ciphertext, c_idx).transpose(0, 1)
            all_losses[batch_start:batch_start + count] = loss_fn(logits).cpu()

    return all_losses

def two_stage_affine_eval(config, ciphertext, loss_fn, K_mult=50, batch_c=512,
                          device=None, verbose=True):
    if device is None:
        device = torch.device("cpu")
    n = len(config.alphabet)
    r = len(config.rotors)
    units = multiplier_units(n)

    mult_combos = list(itertools.product(units, repeat=r))
    P_mult = len(mult_combos)
    s1_combos = [tuple((a, 0) for a in mc) for mc in mult_combos]
    s1_losses = eval_affine_combos(config, s1_combos, ciphertext, loss_fn, batch_c, device)

    K_actual = min(K_mult, P_mult)
    top_k_idxs = s1_losses.argsort()[:K_actual].tolist()
    top_k_mults = [mult_combos[i] for i in top_k_idxs]

    best_loss = float("inf")
    best_combo = None

    for rank_k, mult in enumerate(top_k_mults):
        offset_combos = [
            tuple((mult[i], b_i) for i, b_i in enumerate(bvec))
            for bvec in itertools.product(range(n), repeat=r)
        ]
        losses = eval_affine_combos(
            config, offset_combos, ciphertext, loss_fn, batch_c, device
        )
        local_best_idx = int(losses.argmin())
        local_loss = losses[local_best_idx].item()

        if local_loss < best_loss:
            best_loss = local_loss
            best_combo = offset_combos[local_best_idx]

    return best_combo, best_loss
