import torch
import torch.nn as nn
from .q_net import _make_dft

class ContinuousQRotor(nn.Module):
    def __init__(self, n, F, F_inv, num_candidates=1):
        super().__init__()
        self.n = n
        self.register_buffer('F', F)
        self.register_buffer('F_inv', F_inv)

        P = torch.zeros(n, n)
        for col, row in enumerate(torch.randperm(n).long()):
            P[row, col] = 1.0
        Q = F @ P.to(F.dtype) @ F_inv

        self.Q_real = nn.Parameter(Q.real.contiguous().unsqueeze(0).repeat(num_candidates, 1, 1))
        self.Q_imag = nn.Parameter(Q.imag.contiguous().unsqueeze(0).repeat(num_candidates, 1, 1))

    def get_Q(self):
        return torch.complex(self.Q_real, self.Q_imag)

    def get_spatial_matrix(self):
        return (self.F_inv @ self.get_Q() @ self.F).real

class ContinuousQNet(nn.Module):
    def __init__(self, config, initial_positions):
        super().__init__()
        self.config = config
        self.n = len(config.alphabet)
        self.alphabet = config.alphabet
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.num_rotors = len(config.rotors)

        F, F_inv = _make_dft(self.n)
        self.register_buffer('F', F)
        self.register_buffer('F_inv', F_inv)

        k = torch.arange(self.n, dtype=torch.float32)
        self.register_buffer('k', k)

        self.num_candidates = len(initial_positions)
        pos_init = torch.tensor(
            [[config.parse_position(p) for p in pos_list] for pos_list in initial_positions],
            dtype=torch.long
        )
        self.register_buffer('initial_positions', pos_init)
        self.notches = [config.parse_position(r.notch) for r in config.rotors]

        self.rotors = nn.ModuleList([
            ContinuousQRotor(self.n, self.F, self.F_inv, self.num_candidates)
            for _ in range(self.num_rotors)
        ])

        ref_matrix = torch.from_numpy(config.wiring_to_matrix(config.reflector)).float()
        R_fourier = self.F @ ref_matrix.to(self.F.dtype) @ self.F_inv
        self.register_buffer('R_real', R_fourier.real.contiguous())
        self.register_buffer('R_imag', R_fourier.imag.contiguous())

        self._cached_T = None
        self._cached_step_positions = None

    @property
    def reflector_fourier(self):
        return torch.complex(self.R_real, self.R_imag)

    @property
    def reflector(self):
        return (self.F_inv @ self.reflector_fourier @ self.F).real

    def prune_candidates(self, keep_indices):
        idx = torch.tensor(keep_indices, dtype=torch.long, device=self.initial_positions.device)
        self.initial_positions = self.initial_positions[idx]
        for r in self.rotors:
            r.Q_real = nn.Parameter(r.Q_real.data[idx])
            r.Q_imag = nn.Parameter(r.Q_imag.data[idx])
        self.num_candidates = len(keep_indices)
        self._cached_T = None

    def precompute_steps(self, T):
        if self._cached_T == T:
            return self._cached_step_positions

        positions = self.initial_positions.clone()
        step_positions = []
        notches = torch.tensor(self.notches, dtype=torch.long, device=self.initial_positions.device)
        C = self.num_candidates

        for _ in range(T):
            step = torch.ones(C, dtype=torch.bool, device=self.initial_positions.device)
            for i in range(self.num_rotors - 1, -1, -1):
                at_notch = (positions[:, i] == notches[i])
                positions[:, i] = torch.where(step, (positions[:, i] + 1) % self.n, positions[:, i])
                step = step & at_notch
            step_positions.append(positions.clone())

        self._cached_T = T
        self._cached_step_positions = torch.stack(step_positions, dim=0)
        return self._cached_step_positions

    def forward(self, v, step_position):
        u = self.F @ v.to(self.F.dtype)
        u = u.unsqueeze(0).expand(self.num_candidates, -1)
        for i in range(self.num_rotors - 1, -1, -1):
            Q = self.rotors[i].get_Q()
            phi_eff = step_position[:, i].float()
            phase = torch.exp(-2j * torch.pi * self.k * phi_eff.unsqueeze(-1) / self.n)
            u = torch.bmm((phase * u).unsqueeze(1), Q.transpose(-2, -1)).squeeze(1) * phase.conj()
        u = u @ self.reflector_fourier.T
        for i in range(self.num_rotors):
            Q = self.rotors[i].get_Q()
            phi_eff = step_position[:, i].float()
            phase = torch.exp(-2j * torch.pi * self.k * phi_eff.unsqueeze(-1) / self.n)
            u = torch.bmm((phase * u).unsqueeze(1), Q.conj()).squeeze(1) * phase.conj()
        return (u @ self.F_inv.T).real.float()

    def encrypt_sequence(self, input_indices):
        T = len(input_indices)
        step_positions = self.precompute_steps(T)
        input_indices_t = torch.tensor(input_indices, dtype=torch.long, device=self.initial_positions.device)
        U = self.F[:, input_indices_t].T.unsqueeze(1).expand(-1, self.num_candidates, -1)
        k_expanded = self.k.reshape(1, 1, -1)

        for i in range(self.num_rotors - 1, -1, -1):
            Q = self.rotors[i].get_Q()
            phi_eff = step_positions[:, :, i].float()
            phase = torch.exp(-2j * torch.pi * k_expanded * phi_eff.unsqueeze(-1) / self.n)
            U = torch.einsum("tcn,cni->tci", phase * U, Q.transpose(-2, -1)) * phase.conj()

        U = U @ self.reflector_fourier.T

        for i in range(self.num_rotors):
            Q = self.rotors[i].get_Q()
            phi_eff = step_positions[:, :, i].float()
            phase = torch.exp(-2j * torch.pi * k_expanded * phi_eff.unsqueeze(-1) / self.n)
            U = torch.einsum("tcn,cni->tci", phase * U, Q.conj()) * phase.conj()

        return (U @ self.F_inv.T).real.float()

    def encrypt_sequence_slice(self, input_indices, c_indices, step_positions=None):
        if step_positions is None:
            T = len(input_indices)
            step_positions = self.precompute_steps(T)[:, c_indices, :]

        input_indices_t = torch.tensor(input_indices, dtype=torch.long, device=self.initial_positions.device)
        U = self.F[:, input_indices_t].T.unsqueeze(1).expand(-1, len(c_indices), -1)
        k_expanded = self.k.reshape(1, 1, -1)

        for i in range(self.num_rotors - 1, -1, -1):
            Q = self.rotors[i].get_Q()[c_indices]
            phi_eff = step_positions[:, :, i].float()
            phase = torch.exp(-2j * torch.pi * k_expanded * phi_eff.unsqueeze(-1) / self.n)
            U = torch.einsum("tcn,cni->tci", phase * U, Q.transpose(-2, -1)) * phase.conj()

        U = U @ self.reflector_fourier.T

        for i in range(self.num_rotors):
            Q = self.rotors[i].get_Q()[c_indices]
            phi_eff = step_positions[:, :, i].float()
            phase = torch.exp(-2j * torch.pi * k_expanded * phi_eff.unsqueeze(-1) / self.n)
            U = torch.einsum("tcn,cni->tci", phase * U, Q.conj()) * phase.conj()

        return (U @ self.F_inv.T).real.float()

    def get_positions(self):
        return self.initial_positions.tolist()

    def encrypt_string(self, text, candidate_idx=0, greedy=True):
        indices = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        logits = self.encrypt_sequence(indices)[:, candidate_idx, :]
        if greedy:
            out_indices = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            out_indices = torch.multinomial(probs, 1).squeeze(-1)
        return "".join(self.alphabet[i] for i in out_indices)

def permutation_regularizer(net):
    device = net.rotors[0].Q_real.device
    loss = torch.zeros(net.num_candidates, device=device)
    for rotor in net.rotors:
        P = rotor.get_spatial_matrix()
        row_dev = (P.sum(dim=-1) - 1.0) ** 2
        col_dev = (P.sum(dim=-2) - 1.0) ** 2
        binary = (P * (1.0 - P)).clamp(min=0)
        loss = loss + row_dev.mean(-1) + col_dev.mean(-1) + binary.mean((-1, -2))
    return loss
