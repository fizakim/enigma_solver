import torch
import torch.nn as nn
from ..q_net.net import _make_dft

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
        self.register_buffer('omega', torch.exp(-2j * torch.pi * k / self.n).to(torch.complex64))

        self.num_candidates = len(initial_positions)
        phi_init = torch.tensor(
            [[float(config.parse_position(p)) for p in pos_list] for pos_list in initial_positions],
            dtype=torch.float32
        )
        self.phi = nn.Parameter(phi_init)
        self.notches = [config.parse_position(r.notch) for r in config.rotors]

        self.rotors = nn.ModuleList([
            ContinuousQRotor(self.n, self.F, self.F_inv, self.num_candidates)
            for _ in range(self.num_rotors)
        ])

        ref_matrix = torch.from_numpy(config.wiring_to_matrix(config.reflector)).float()
        R_fourier = self.F @ ref_matrix.to(self.F.dtype) @ self.F_inv
        self.register_buffer('R_real', R_fourier.real.contiguous())
        self.register_buffer('R_imag', R_fourier.imag.contiguous())

        self._cached_int_positions = None
        self._cached_step_offsets = None
        self._cached_T = None

    @property
    def reflector_fourier(self):
        return torch.complex(self.R_real, self.R_imag)

    @property
    def reflector(self):
        return (self.F_inv @ self.reflector_fourier @ self.F).real

    def prune_candidates(self, keep_indices):
        idx = torch.tensor(keep_indices, dtype=torch.long, device=self.phi.device)
        self.phi = nn.Parameter(self.phi.data[idx])
        for r in self.rotors:
            r.Q_real = nn.Parameter(r.Q_real.data[idx])
            r.Q_imag = nn.Parameter(r.Q_imag.data[idx])
        self.num_candidates = len(keep_indices)

    def precompute_steps(self, T):
        int_positions = torch.round(self.phi.detach()).long() % self.n
        if (self._cached_int_positions is not None and 
            self._cached_T == T and 
            torch.equal(int_positions, self._cached_int_positions)):
            return self._cached_step_offsets

        positions = int_positions.clone()
        offsets = []
        notches = torch.tensor(self.notches, dtype=torch.long, device=self.phi.device)
        C = self.phi.shape[0]

        for _ in range(T):
            step = torch.ones(C, dtype=torch.bool, device=self.phi.device)
            for i in range(self.num_rotors - 1, -1, -1):
                at_notch = (positions[:, i] == notches[i])
                positions[:, i] = torch.where(step, (positions[:, i] + 1) % self.n, positions[:, i])
                step = step & at_notch
            offsets.append((positions - int_positions) % self.n)

        self._cached_int_positions = int_positions.clone()
        self._cached_T = T
        self._cached_step_offsets = torch.stack(offsets, dim=0)
        return self._cached_step_offsets

    def forward(self, v, step_offset):
        u = self.F @ v.to(self.F.dtype)
        u = u.unsqueeze(0).expand(self.phi.shape[0], -1)
        for i in range(self.num_rotors - 1, -1, -1):
            Q = self.rotors[i].get_Q()
            phi_eff = self.phi[:, i] + step_offset[:, i].float()
            phase = torch.exp(-2j * torch.pi * self.k * phi_eff.unsqueeze(-1) / self.n)
            u = torch.bmm((phase * u).unsqueeze(1), Q.transpose(-2, -1)).squeeze(1) * phase.conj()
        u = u @ self.reflector_fourier.T
        for i in range(self.num_rotors):
            Q = self.rotors[i].get_Q()
            phi_eff = self.phi[:, i] + step_offset[:, i].float()
            phase = torch.exp(-2j * torch.pi * self.k * phi_eff.unsqueeze(-1) / self.n)
            u = torch.bmm((phase * u).unsqueeze(1), Q.conj()).squeeze(1) * phase.conj()
        return (u @ self.F_inv.T).real.float()

    def encrypt_sequence(self, input_indices):
        T = len(input_indices)
        step_offsets = self.precompute_steps(T)
        input_indices_t = torch.tensor(input_indices, dtype=torch.long, device=self.phi.device)
        U = self.F[:, input_indices_t].T.unsqueeze(1).expand(-1, self.phi.shape[0], -1)
        k_expanded = self.k.reshape(1, 1, -1)

        for i in range(self.num_rotors - 1, -1, -1):
            Q = self.rotors[i].get_Q()
            phi_eff = self.phi[:, i].unsqueeze(0) + step_offsets[:, :, i].float()
            phase = torch.exp(-2j * torch.pi * k_expanded * phi_eff.unsqueeze(-1) / self.n)
            U = torch.einsum("tcn,cni->tci", phase * U, Q.transpose(-2, -1)) * phase.conj()

        U = U @ self.reflector_fourier.T

        for i in range(self.num_rotors):
            Q = self.rotors[i].get_Q()
            phi_eff = self.phi[:, i].unsqueeze(0) + step_offsets[:, :, i].float()
            phase = torch.exp(-2j * torch.pi * k_expanded * phi_eff.unsqueeze(-1) / self.n)
            U = torch.einsum("tcn,cni->tci", phase * U, Q.conj()) * phase.conj()

        return (U @ self.F_inv.T).real.float()

    def get_positions(self):
        pos = self.phi.detach()
        return pos[0].tolist() if pos.shape[0] == 1 else pos.tolist()

    def get_integer_positions(self):
        int_pos = torch.round(self.phi.detach()).long() % self.n
        return int_pos[0].tolist() if int_pos.shape[0] == 1 else int_pos.tolist()

    def encrypt_string(self, text, candidate_idx=0, greedy=True):
        indices = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        logits = self.encrypt_sequence(indices)[:, candidate_idx, :]
        if greedy:
            out_indices = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            out_indices = torch.multinomial(probs, 1).squeeze(-1)
        return "".join(self.alphabet[i] for i in out_indices)
