import torch
import torch.nn as nn
from .q_net import _make_dft


class ContinuousPositionNet(nn.Module):
    """
    Dual of ContinuousQNet: fixed rotor wirings, learnable continuous positions φ.

    Given known rotor/reflector wirings (from config), optimises the initial
    position vector φ ∈ ℝ^r to minimise the n-gram (or IC) loss on ciphertext.

    Mathematical structure
    ----------------------
    For candidate c and rotor i at time step t the effective position is:

        φ_eff(t, c, i)  =  step_pos(t, c, i)  +  {φ[c, i]}

    where
        step_pos(t, c, i)  ∈ ℤ   — integer Enigma step from round(φ[c]) (no gradient)
        {φ[c, i]}  =  φ[c, i] − round(φ[c, i])   — fractional part (gradient path)

    Gradient chain: ∂L/∂φ[c,i] = ∂L/∂φ_eff · (∂φ_eff/∂φ) = ∂L/∂φ_eff · 1.

    The DFT phase D_{-φ_eff}[k] = exp(−2πi k φ_eff / n) is smooth in φ_eff, giving
    a clean gradient through the commutator [D_G, D_φ Q D_{-φ}] (D_G = diag(−2πik/n)).

    Basin structure
    ---------------
    With Enigma stepping the IC loss is uniquely minimised at the true position φ*.
    The gradient is strong within [φ*_i ± 0.5] for each rotor i (Dirichlet smearing
    is maximal at half-integers, collapsing IC to near-random outside the basin).
    Multi-start at all n^r integer positions guarantees at least one candidate starts
    in the correct basin.

    Efficiency advantage over ContinuousQNet
    -----------------------------------------
    Q is shared across all candidates, so the matmul is [T,C,n] @ [n,n] rather than
    the per-candidate [T,C,n] @ [C,n,n] einsum — cheaper memory and FLOPs.
    """

    def __init__(self, config, initial_phi):
        """
        config      : EnigmaConfig — provides fixed wirings for rotors and reflector
        initial_phi : Tensor [C, r] or list-of-lists of real initial positions
        """
        super().__init__()
        self.n = len(config.alphabet)
        self.alphabet = config.alphabet
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.num_rotors = len(config.rotors)

        F, F_inv = _make_dft(self.n)
        self.register_buffer('F', F)
        self.register_buffer('F_inv', F_inv)

        k = torch.arange(self.n, dtype=torch.float32)
        self.register_buffer('k', k)

        # Fixed DFT-domain rotor matrices  Q_i = F P_i F⁻¹  (shared across candidates)
        self._q_buf_names = []
        for i, r_cfg in enumerate(config.rotors):
            P = torch.from_numpy(config.wiring_to_matrix(r_cfg.wiring)).float()
            Q = F @ P.to(F.dtype) @ F_inv
            name = f'_Q_r{i}'
            self.register_buffer(name, Q.contiguous())
            self._q_buf_names.append(name)

        # Fixed DFT-domain reflector  Q_R = F R F⁻¹
        R = torch.from_numpy(config.wiring_to_matrix(config.reflector)).float()
        self.register_buffer('_Q_refl', (F @ R.to(F.dtype) @ F_inv).contiguous())

        # Notch positions for Enigma stepping
        self.register_buffer(
            'notches',
            torch.tensor(
                [config.parse_position(rc.notch) for rc in config.rotors],
                dtype=torch.long,
            ),
        )

        # Learnable initial positions  φ ∈ ℝ^{C × r}
        if not isinstance(initial_phi, torch.Tensor):
            initial_phi = torch.tensor(initial_phi, dtype=torch.float32)
        self.phi = nn.Parameter(initial_phi.float())
        self.num_candidates = self.phi.shape[0]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_Q(self, i):
        """Return [n, n] complex Q matrix for rotor i (shared across candidates)."""
        return getattr(self, self._q_buf_names[i])

    def _precompute_steps(self, round_phi, T):
        """
        Compute integer effective positions for each candidate and time step.

        Implements the Enigma carry-propagation stepping (including double-stepping):
          - rightmost rotor always steps
          - carry propagates left when a rotor is at its notch

        round_phi : [C, r] long   — rounded initial positions
        T         : int           — sequence length
        Returns   : [T, C, r] long
        """
        device = round_phi.device
        C = round_phi.shape[0]
        n = self.n
        notches = self.notches.to(device)

        positions = round_phi.clone()
        steps = []
        for _ in range(T):
            carry = torch.ones(C, dtype=torch.bool, device=device)
            for i in range(self.num_rotors - 1, -1, -1):
                at_notch = positions[:, i] == notches[i]
                positions[:, i] = torch.where(
                    carry, (positions[:, i] + 1) % n, positions[:, i]
                )
                carry = carry & at_notch
            steps.append(positions.clone())

        return torch.stack(steps)  # [T, C, r]

    def _phase(self, phi_eff_i):
        """
        phi_eff_i : [T, C]  real effective position for one rotor
        Returns   : [T, C, n] complex  D_{-phi_eff_i}[k] = exp(−2πi k φ / n)
        """
        return torch.exp(
            -2j * torch.pi
            * self.k[None, None, :]        # [1, 1, n]
            * phi_eff_i[:, :, None]        # [T, C, 1]
            / self.n
        )

    # ------------------------------------------------------------------ #
    # Forward pass                                                          #
    # ------------------------------------------------------------------ #

    def encrypt_sequence_slice(self, input_indices, c_indices):
        """
        Differentiable forward pass for a subset of candidates.

        input_indices : list[int]  length T
        c_indices     : 1-D LongTensor  length C_batch
        Returns       : [T, C_batch, n] real logits
        """
        T = len(input_indices)
        device = self.phi.device

        if not isinstance(c_indices, torch.Tensor):
            c_indices = torch.tensor(c_indices, dtype=torch.long, device=device)
        C_b = c_indices.shape[0]

        phi_b = self.phi[c_indices]                           # [C_b, r]
        round_b = phi_b.detach().round().long() % self.n     # [C_b, r]  no gradient
        frac_b  = phi_b - phi_b.detach().round()             # [C_b, r]  gradient path

        step_pos = self._precompute_steps(round_b, T)        # [T, C_b, r]
        phi_eff  = step_pos.float() + frac_b.unsqueeze(0)    # [T, C_b, r]

        input_t = torch.tensor(input_indices, dtype=torch.long, device=device)
        U = self.F[:, input_t].T.to(torch.complex64)         # [T, n]
        U = U.unsqueeze(1).expand(-1, C_b, -1).clone()       # [T, C_b, n]

        # Forward rotors (rightmost first — Enigma convention)
        for i in range(self.num_rotors - 1, -1, -1):
            Q     = self._get_Q(i)                           # [n, n] complex
            phase = self._phase(phi_eff[:, :, i])            # [T, C_b, n]
            U = (phase * U) @ Q.T                            # [T, C_b, n]
            U = phase.conj() * U

        # Reflector
        U = U @ self._Q_refl.T                               # [T, C_b, n]

        # Backward rotors (leftmost first)
        for i in range(self.num_rotors):
            Q     = self._get_Q(i)
            phase = self._phase(phi_eff[:, :, i])
            U = (phase * U) @ Q.conj()                       # [T, C_b, n]
            U = phase.conj() * U

        return (U @ self.F_inv.T).real.float()               # [T, C_b, n]

    def encrypt_sequence(self, input_indices):
        """Forward pass for all candidates."""
        c_all = torch.arange(self.num_candidates, device=self.phi.device)
        return self.encrypt_sequence_slice(input_indices, c_all)

    # ------------------------------------------------------------------ #
    # Utilities                                                             #
    # ------------------------------------------------------------------ #

    def prune_candidates(self, keep_indices):
        idx = torch.tensor(keep_indices, dtype=torch.long, device=self.phi.device)
        self.phi = nn.Parameter(self.phi.data[idx])
        self.num_candidates = len(keep_indices)

    def get_integer_positions(self):
        """Return round(φ) % n for each candidate as a list-of-lists."""
        return (self.phi.detach().round().long() % self.n).tolist()

    def get_continuous_positions(self):
        return self.phi.detach().tolist()

    def encrypt_string(self, text, candidate_idx=0, greedy=True):
        indices = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        c_idx = torch.tensor([candidate_idx], dtype=torch.long, device=self.phi.device)
        logits = self.encrypt_sequence_slice(indices, c_idx)[:, 0, :]
        if greedy:
            out = torch.argmax(logits, dim=-1)
        else:
            out = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
        return ''.join(self.alphabet[i] for i in out)
