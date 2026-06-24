"""Wiring-basin instances at a fixed, known rotor position.

``basin_instances(config, positions)`` builds the requested
``basin_instances(p) = [Q_net_1, Q_net_2, ...]``: ``k = phi(n)^r`` wiring networks,
all pinned to the single known ``positions`` and each initialised at a distinct
multiplier anchor (the Q-basin). They are bundled into one batched ``ContinuousQNet``
(candidate axis = basin index) so the whole list trains in parallel.

Position is held in a buffer here and is never an optimiser parameter — training
only moves the per-basin wirings ``Q``.
"""

import itertools

import torch

from .net import ContinuousQNet
from ..affine import multiplier_units, multiplier_anchor_Q, affine_anchor_Q

__all__ = ["basin_instances", "affine_basin_instances", "stream_affine_eval",
           "stream_affine_combo", "single_rotor_affine_score",
           "eval_affine_combos", "two_stage_affine_eval"]


def basin_instances(config, positions):
    """Return a batched ``ContinuousQNet`` of ``k = phi(n)^r`` wiring basins.

    Every candidate is fixed at ``positions`` (the known rotor position) and rotor ``i``
    of candidate ``c`` is initialised at the multiplier anchor for that basin's tuple.
    The chosen multiplier tuples are stored on ``net.basin_combos``.
    """
    n = len(config.alphabet)
    r = len(config.rotors)
    combos = list(itertools.product(multiplier_units(n), repeat=r))  # k basins

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
    """Return a batched ``ContinuousQNet`` of ``k = (phi(n)*n)^r`` affine wiring basins.

    Each rotor of each candidate is initialised at the affine anchor Q_{a,b} for its
    ``(a, b)`` pair.  All candidates start at position 0 — the initial position is fully
    absorbed into the wiring offset ``b`` (b = (a-1)*phi mod n for the equivalent
    multiplier-at-phi config).  No ``positions`` argument is needed or accepted.
    """
    n = len(config.alphabet)
    r = len(config.rotors)
    per_rotor = list(itertools.product(multiplier_units(n), range(n)))  # phi(n)*n pairs
    combos = list(itertools.product(per_rotor, repeat=r))               # (phi(n)*n)^r

    # Memory estimate: C * r * n^2 * 8 bytes (two float32 [n,n] matrices per rotor).
    mem_bytes = len(combos) * r * n * n * 8
    if mem_bytes > 4 * 1024**3:  # 4 GB soft limit
        import warnings
        warnings.warn(
            f"affine_basin_instances: {len(combos):,} candidates × {r} rotors × {n}² "
            f"≈ {mem_bytes / 1024**3:.1f} GB. "
            "Consider a smaller alphabet or hierarchical search for n≥26."
        )

    net = ContinuousQNet(config, initial_positions=[[0] * r for _ in combos])
    F, F_inv = net.F, net.F_inv
    with torch.no_grad():
        for c, combo in enumerate(combos):
            for i, (a, b) in enumerate(combo):
                Qab = affine_anchor_Q(a, b, F, F_inv)
                net.rotors[i].Q_real.data[c] = Qab.real
                net.rotors[i].Q_imag.data[c] = Qab.imag

    net.basin_combos = combos  # each entry: tuple of (a, b) per rotor
    return net


def single_rotor_affine_score(config, rotor_idx, fixed_wirings, ciphertext, loss_fn,
                              device=None):
    """Score all phi(n)*n affine wirings for one rotor while others are held fixed.

    Inner step of coordinate descent: fix all rotors except ``rotor_idx`` and score
    every valid affine wiring ``(a, b)`` for that rotor.  The language loss is only
    ever evaluated on valid Enigma decryptions — no cheating possible.

    Args:
        config: EnigmaConfig.
        rotor_idx: index of the rotor to vary (0 .. r-1).
        fixed_wirings: list of ``(a, b)`` tuples length ``r``; entry at ``rotor_idx``
            is ignored.
        ciphertext: list of int indices (encrypted input).
        loss_fn: callable ``[B, T, n] -> [B]``.
        device: torch device (default: CPU).

    Returns:
        ``(per_rotor, losses)`` where ``per_rotor`` is the list of ``phi(n)*n`` ``(a,b)``
        pairs and ``losses`` is a CPU float32 tensor of shape ``[P]``.
        ``per_rotor[int(losses.argmin())]`` gives the best wiring for this rotor.
    """
    if device is None:
        device = torch.device("cpu")

    n = len(config.alphabet)
    r = len(config.rotors)
    per_rotor = list(itertools.product(multiplier_units(n), range(n)))  # P = phi(n)*n
    P = len(per_rotor)

    # F, F_inv from a throw-away single-candidate net.
    _tmp = ContinuousQNet(config, initial_positions=[[0] * r])
    F, F_inv = _tmp.F, _tmp.F_inv
    del _tmp

    # Pre-compute the P varying-rotor Q matrices on CPU then move to device.
    Q_real_var = torch.empty(P, n, n)
    Q_imag_var = torch.empty(P, n, n)
    for c, (a, b) in enumerate(per_rotor):
        Qab = affine_anchor_Q(a, b, F, F_inv)
        Q_real_var[c] = Qab.real
        Q_imag_var[c] = Qab.imag
    Q_real_var = Q_real_var.to(device)
    Q_imag_var = Q_imag_var.to(device)

    # Build a P-candidate net and fill it.
    net = ContinuousQNet(config, initial_positions=[[0] * r] * P).to(device)
    with torch.no_grad():
        # Fixed rotors: all P candidates share the same Q (broadcast via slice assign).
        for j in range(r):
            if j == rotor_idx:
                continue
            a_j, b_j = fixed_wirings[j]
            Qj = affine_anchor_Q(a_j, b_j, F, F_inv).to(device)
            net.rotors[j].Q_real.data[:] = Qj.real
            net.rotors[j].Q_imag.data[:] = Qj.imag

        # Varying rotor: candidate c gets wiring per_rotor[c].
        net.rotors[rotor_idx].Q_real.data[:] = Q_real_var
        net.rotors[rotor_idx].Q_imag.data[:] = Q_imag_var

        c_idx = torch.arange(P, device=device)
        logits = net.encrypt_sequence_slice(ciphertext, c_idx).transpose(0, 1)
        losses = loss_fn(logits).cpu()

    return per_rotor, losses


def stream_affine_eval(config, ciphertext, loss_fn, batch_c=512, device=None):
    """Evaluate all ``(phi(n)*n)^r`` affine combos without pre-allocating all Q matrices.

    Designed for n=26 where pre-allocation would require tens of GB.  Instead:

    1. Pre-computes a Q-table of all ``phi(n)*n`` single-rotor affine matrices — tiny
       (e.g. 312 × 26 × 26 for n=26, ~17 MB).
    2. Creates a reusable ``ContinuousQNet`` of size ``batch_c``.
    3. Streams through all ``(phi(n)*n)^r`` combos, filling Q matrices from the table
       and evaluating ``batch_c`` candidates at a time.

    Args:
        config: EnigmaConfig with alphabet and rotor specs.
        ciphertext: list of int indices (the encrypted input).
        loss_fn: callable [B,T,n] → [B] (e.g. NgramLoss or TransformerLoss).
        batch_c: candidates evaluated per iteration.
        device: torch device.

    Returns:
        ``(per_rotor, losses)`` where ``per_rotor`` is the list of ``phi(n)*n``
        single-rotor ``(a, b)`` pairs (index i decodes as ``per_rotor[i]``), and
        ``losses`` is a float32 tensor of shape ``[P^r]`` containing the loss for
        every combo in lexicographic order over ``per_rotor`` indices.  Use
        ``stream_affine_combo(idx, per_rotor, r)`` to decode an argmin index.
    """
    if device is None:
        device = torch.device("cpu")

    n = len(config.alphabet)
    r = len(config.rotors)
    units = multiplier_units(n)
    per_rotor = list(itertools.product(units, range(n)))  # P = phi(n)*n entries
    P = len(per_rotor)

    # --- pre-compute Q-table for all single-rotor (a, b) pairs ---------------
    _dummy = ContinuousQNet(config, initial_positions=[[0] * r])
    F, F_inv = _dummy.F, _dummy.F_inv
    del _dummy

    Q_real_tbl = torch.empty(P, n, n)
    Q_imag_tbl = torch.empty(P, n, n)
    for idx, (a, b) in enumerate(per_rotor):
        Qab = affine_anchor_Q(a, b, F, F_inv)
        Q_real_tbl[idx] = Qab.real
        Q_imag_tbl[idx] = Qab.imag
    Q_real_tbl = Q_real_tbl.to(device)
    Q_imag_tbl = Q_imag_tbl.to(device)

    # --- reusable net of size batch_c (positions all 0) -----------------------
    net = ContinuousQNet(config, initial_positions=[[0] * r] * batch_c).to(device)

    C_total = P ** r
    all_losses = torch.empty(C_total)

    # Rotor-index workspace: shape [batch_c, r].  Re-filled each iteration via
    # vectorised flat-index arithmetic — avoids 30M Python-level iterations.
    ri = torch.empty(batch_c, r, dtype=torch.long, device=device)
    _c_idx = torch.arange(batch_c, device=device)  # reused every _flush call

    def _flush(count):
        """Evaluate the first ``count`` candidates in the reusable net."""
        with torch.no_grad():
            for rot_i in range(r):
                net.rotors[rot_i].Q_real.data[:count] = Q_real_tbl[ri[:count, rot_i]]
                net.rotors[rot_i].Q_imag.data[:count] = Q_imag_tbl[ri[:count, rot_i]]
            logits = net.encrypt_sequence_slice(ciphertext, _c_idx[:count]).transpose(0, 1)
            return loss_fn(logits).cpu()

    n_batches = (C_total + batch_c - 1) // batch_c
    log_every = max(1, n_batches // 20)   # print ~20 progress updates total

    for batch_num, batch_start in enumerate(range(0, C_total, batch_c)):
        count = min(batch_c, C_total - batch_start)
        # Decode flat indices [batch_start, batch_start+count) into per-rotor indices.
        temp = torch.arange(batch_start, batch_start + count,
                            dtype=torch.long, device=device)
        for rot_i in range(r - 1, -1, -1):
            ri[:count, rot_i] = temp % P
            temp = temp // P

        all_losses[batch_start:batch_start + count] = _flush(count)

        if batch_num % log_every == 0 or batch_num == n_batches - 1:
            pct = 100.0 * (batch_start + count) / C_total
            print(f"  {pct:5.1f}%  ({batch_start + count:>{len(str(C_total))},}/{C_total:,})",
                  flush=True)

    return per_rotor, all_losses


def stream_affine_combo(flat_idx, per_rotor, r):
    """Decode a flat index (e.g. argmin of losses) back to a combo tuple.

    Returns a tuple of ``r`` ``(a, b)`` pairs corresponding to ``flat_idx`` in the
    lexicographic ordering produced by ``stream_affine_eval``.
    """
    P = len(per_rotor)
    combo = []
    for _ in range(r):
        combo.append(per_rotor[flat_idx % P])
        flat_idx //= P
    return tuple(reversed(combo))


def eval_affine_combos(config, combos, ciphertext, loss_fn, batch_c=512, device=None):
    """Evaluate a specific list of affine combo configs without enumerating the full space.

    Uses the same pre-computed Q-table trick as ``stream_affine_eval`` but evaluates
    only the combos provided rather than all ``P^r``.

    Args:
        combos: list of r-tuples, each tuple = r ``(a_i, b_i)`` pairs (one per rotor).
                All ``a_i`` must satisfy ``gcd(a_i, n) == 1``.
        ciphertext: list of int indices (the encrypted input).
        loss_fn: callable ``[B, T, n] -> [B]``.
        batch_c: candidates evaluated per batch.
        device: torch device.

    Returns:
        float32 tensor of shape ``[len(combos)]`` with the loss for each combo.
    """
    if device is None:
        device = torch.device("cpu")
    n = len(config.alphabet)
    r = len(config.rotors)

    per_rotor = list(itertools.product(multiplier_units(n), range(n)))
    ab_to_idx = {ab: i for i, ab in enumerate(per_rotor)}

    _dummy = ContinuousQNet(config, initial_positions=[[0] * r])
    F, F_inv = _dummy.F, _dummy.F_inv
    del _dummy

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
    """Two-stage affine wiring search — 34× faster than brute force for n=26.

    The multiplier ``a`` and offset ``b`` of each affine wiring ``x → ax+b`` contribute
    independently to the Fourier Q-matrix (``Q_{a,b} = diag(phase_b) · Q_a``).  This
    structure enables a two-stage decomposition:

    **Stage 1 — Multiplier scan** (``phi(n)^r`` evals, e.g. 1,728 for n=26):
        Evaluate all multiplier combos with ``b=0``.  Even without the true offsets,
        the correct multiplier tuple produces better language statistics than wrong
        multipliers because the multiplier governs the fundamental permutation structure
        while offsets only shift characters cyclically.  Keep the top-``K_mult``.

    **Stage 2 — Offset flood-fill** (``K_mult × n^r`` evals, e.g. 50 × 17,576):
        For each top-K multiplier combo, exhaustively evaluate all ``n^r`` offset combos.
        With correct multipliers, the language signal is unambiguous — this succeeds
        where coordinate descent (``coordinate_descent_train.py``) fails because it never
        has to search with two wrong rotors simultaneously.

    **Total for n=26, r=3, K_mult=50**:
        ~1,728 + 50 × 17,576 = ~880K evaluations vs 30M flat search (34× speedup).
        Runs on CPU. No GPU required.

    Args:
        K_mult: number of multiplier combos to carry into Stage 2.  Larger K_mult
                raises reliability (true multiplier in top-K) at linear cost.
        batch_c: candidates per batch for the inner ``eval_affine_combos`` calls.
        device: torch device (CPU works fine).
        verbose: print progress.

    Returns:
        ``(best_combo, best_loss)`` where ``best_combo`` is an r-tuple of ``(a, b)``
        pairs (one per rotor) and ``best_loss`` is its language loss.
    """
    if device is None:
        device = torch.device("cpu")
    n = len(config.alphabet)
    r = len(config.rotors)
    units = multiplier_units(n)

    # --- Stage 1: multiplier scan (b=0 for every rotor) ----------------------
    mult_combos = list(itertools.product(units, repeat=r))
    P_mult = len(mult_combos)
    s1_combos = [tuple((a, 0) for a in mc) for mc in mult_combos]
    if verbose:
        print(f"Stage 1: {P_mult} multiplier combos (b=0)...")
    s1_losses = eval_affine_combos(config, s1_combos, ciphertext, loss_fn, batch_c, device)

    K_actual = min(K_mult, P_mult)
    top_k_idxs = s1_losses.argsort()[:K_actual].tolist()
    top_k_mults = [mult_combos[i] for i in top_k_idxs]
    if verbose:
        print(f"  Stage 1 done. Top-{K_actual} multiplier combos kept "
              f"(best loss={s1_losses[top_k_idxs[0]]:.4f})")

    # --- Stage 2: offset flood-fill for each top-K multiplier ----------------
    n_offsets = n ** r
    total_s2 = K_actual * n_offsets
    if verbose:
        print(f"Stage 2: {K_actual} × {n_offsets:,} = {total_s2:,} offset evals...")
    log_every = max(1, K_actual // 10)

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

        if verbose and (rank_k % log_every == 0 or rank_k == K_actual - 1):
            print(f"  [{rank_k + 1:>{len(str(K_actual))}}/{K_actual}] "
                  f"mult={mult}  best_offset_loss={local_loss:.4f}  "
                  f"global_best={best_loss:.4f}")

    return best_combo, best_loss
