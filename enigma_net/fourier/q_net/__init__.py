import torch
import torch.nn as nn

def _make_dft(n):
    k = torch.arange(n, dtype=torch.float64)
    F = torch.exp(-2j * torch.pi * k.unsqueeze(0) * k.unsqueeze(1) / n) / (n ** 0.5)
    return F.to(torch.complex64), F.conj().T.contiguous().to(torch.complex64)

def _perm_to_matrix(perm_indices, n):
    P = torch.zeros(n, n)
    for col, row in enumerate(perm_indices.long()):
        P[row, col] = 1.0
    return P

class QRotor(nn.Module):
    def __init__(self, n, F, F_inv, target_wiring=None):
        super().__init__()
        self.n = n
        self.register_buffer('F', F)
        self.register_buffer('F_inv', F_inv)

        if target_wiring is not None:
            Q = F @ target_wiring.to(F.dtype) @ F_inv
            self.register_buffer('Q_real', Q.real.contiguous())
            self.register_buffer('Q_imag', Q.imag.contiguous())
        else:
            perm_idx = torch.randperm(n)
            P = _perm_to_matrix(perm_idx, n)
            Q = F @ P.to(F.dtype) @ F_inv
            self.Q_real = nn.Parameter(Q.real.contiguous())
            self.Q_imag = nn.Parameter(Q.imag.contiguous())

    def get_Q(self):
        return torch.complex(self.Q_real, self.Q_imag)

    def get_spatial_matrix(self):
        return (self.F_inv @ self.get_Q() @ self.F).real

class QNet(nn.Module):
    def __init__(self, config, load_target=False, trainable_rotors=None, trainable_reflector=False):
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

        self.rotors = nn.ModuleList([
            QRotor(
                self.n, F, F_inv,
                target_wiring=torch.from_numpy(config.wiring_to_matrix(r.wiring)).float() if load_target else None
            )
            for r in config.rotors
        ])

        ref_matrix = torch.from_numpy(config.wiring_to_matrix(config.reflector)).float()
        R_fourier = F @ ref_matrix.to(F.dtype) @ F_inv

        if trainable_reflector and not load_target:
            self.R_real = nn.Parameter(R_fourier.real.contiguous())
            self.R_imag = nn.Parameter(R_fourier.imag.contiguous())
        else:
            self.register_buffer('R_real', R_fourier.real.contiguous())
            self.register_buffer('R_imag', R_fourier.imag.contiguous())

        self.notches = [config.parse_position(r.notch) for r in config.rotors]
        self.reset()

        if load_target:
            for p in self.parameters():
                p.requires_grad = False
        elif trainable_rotors is not None:
            for i, rotor in enumerate(self.rotors):
                if i not in trainable_rotors:
                    for p in rotor.parameters():
                        p.requires_grad = False

    @property
    def reflector_fourier(self):
        return torch.complex(self.R_real, self.R_imag)

    @property
    def reflector(self):
        return (self.F_inv @ self.reflector_fourier @ self.F).real

    def reset(self, positions=None):
        if positions is None:
            positions = [0] * self.num_rotors
        self.positions = [self.config.parse_position(p) for p in positions]

    def step(self):
        for i in range(self.num_rotors - 1, -1, -1):
            at_notch = self.positions[i] == self.notches[i]
            self.positions[i] = (self.positions[i] + 1) % self.n
            if not at_notch:
                break

    def get_spatial_matrix(self, rotor_idx):
        return self.rotors[rotor_idx].get_spatial_matrix()

    def _apply_forward_rotors(self, u):
        for rotor, pos in zip(reversed(self.rotors), reversed(self.positions)):
            Q = rotor.get_Q()
            phase = torch.exp(-2j * torch.pi * self.k * pos / self.n).to(torch.complex64)
            u = phase * u
            u = Q @ u
            u = phase.conj() * u
        return u

    def _apply_backward_rotors(self, u):
        for rotor, pos in zip(self.rotors, self.positions):
            Q = rotor.get_Q()
            phase = torch.exp(-2j * torch.pi * self.k * pos / self.n).to(torch.complex64)
            u = phase * u
            u = Q.conj().mT @ u
            u = phase.conj() * u
        return u

    def forward(self, v):
        self.step()
        u = self.F @ v.to(self.F.dtype)
        u = self._apply_forward_rotors(u)
        u = self.reflector_fourier @ u
        u = self._apply_backward_rotors(u)
        return (self.F_inv @ u).real.float()

    def forward_matrix(self, positions=None):
        if positions is not None:
            self.reset(positions)
        self.step()

        eye = torch.eye(self.n)
        cols = []
        for j in range(self.n):
            v = eye[:, j]
            u = self.F @ v.to(self.F.dtype)
            u = self._apply_forward_rotors(u)
            u = self.reflector_fourier @ u
            u = self._apply_backward_rotors(u)
            cols.append((self.F_inv @ u).real.float())
        return torch.stack(cols, dim=1)

    def encrypt_sequence(self, input_indices):
        T = len(input_indices)
        device = self.F.device

        positions = list(self.positions)
        step_pos = torch.empty(T, self.num_rotors, dtype=torch.long, device=device)
        for t in range(T):
            for i in range(self.num_rotors - 1, -1, -1):
                at_notch = (positions[i] == self.notches[i])
                positions[i] = (positions[i] + 1) % self.n
                if not at_notch:
                    break
            for r in range(self.num_rotors):
                step_pos[t, r] = positions[r]

        input_t = torch.tensor(input_indices, dtype=torch.long, device=device)
        U = self.F[:, input_t].T

        for i in range(self.num_rotors - 1, -1, -1):
            Q = self.rotors[i].get_Q()
            phi = step_pos[:, i].float().unsqueeze(1)
            phase = torch.exp(-2j * torch.pi * self.k.unsqueeze(0) * phi / self.n)
            U = (phase * U) @ Q.T
            U = phase.conj() * U

        U = U @ self.reflector_fourier.T

        for i in range(self.num_rotors):
            Q = self.rotors[i].get_Q()
            phi = step_pos[:, i].float().unsqueeze(1)
            phase = torch.exp(-2j * torch.pi * self.k.unsqueeze(0) * phi / self.n)
            U = (phase * U) @ Q.conj()
            U = phase.conj() * U

        return (U @ self.F_inv.T).real.float()

    def encrypt_string(self, text, greedy=False):
        res = []
        for c in text:
            if c not in self.char_to_idx:
                res.append(c)
                continue
            v = torch.zeros(self.n)
            v[self.char_to_idx[c]] = 1.0
            out = torch.softmax(self.forward(v), dim=-1)
            idx = torch.argmax(out).item() if greedy else torch.multinomial(out, 1).item()
            res.append(self.alphabet[idx])
        return "".join(res)
