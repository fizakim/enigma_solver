import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

def _dft_matrix(n):
    k = np.arange(n).reshape(n, 1)
    m = np.arange(n).reshape(1, n)
    return np.exp(-2j * np.pi * k * m / n) / np.sqrt(n)

def visualise_q_net(net, target_sim, position=None, show_numbers=True):
    alphabet = net.config.alphabet
    n = len(alphabet)
    F = _dft_matrix(n)
    F_inv = F.conj().T

    if position is None:
        position = [0] * net.num_rotors

    net.reset(position)
    net.step()

    target_sim.reset(position)
    for r in reversed(target_sim.rotors):
        if not r.step():
            break

    n_rotors = net.num_rotors
    rotor_data = []
    for i, (net_rotor, tgt_rotor) in enumerate(zip(net.rotors, target_sim.rotors)):
        pos = tgt_rotor.position

        W = torch.from_numpy(tgt_rotor.matrix).float()
        tgt_spatial = torch.roll(W, shifts=(-pos, -pos), dims=(0, 1)).numpy()

        tgt_Q_fourier = F @ tgt_spatial @ F_inv
        tgt_Q_mag = np.abs(tgt_Q_fourier)

        with torch.no_grad():
            lrn_spatial = net_rotor.get_spatial_matrix().numpy()
            lrn_Q = net_rotor.get_Q().detach().cpu().numpy()
        lrn_Q_mag = np.abs(lrn_Q)

        argmax_spatial = np.zeros_like(lrn_spatial)
        argmax_idx = np.argmax(lrn_spatial, axis=0)
        argmax_spatial[argmax_idx, np.arange(n)] = 1.0

        rotor_data.append({
            "label": f"Rotor {i}  (pos={pos})",
            "tgt_spatial": tgt_spatial,
            "lrn_spatial": lrn_spatial,
            "argmax_spatial": argmax_spatial,
            "tgt_Q_mag": tgt_Q_mag,
            "lrn_Q_mag": lrn_Q_mag,
        })

    tgt_ref_spatial = target_sim.reflector.matrix
    tgt_ref_fourier = F @ tgt_ref_spatial @ F_inv
    tgt_ref_mag = np.abs(tgt_ref_fourier)

    with torch.no_grad():
        lrn_ref_spatial = net.reflector.detach().cpu().numpy()
        lrn_ref_fourier = net.reflector_fourier.detach().cpu().numpy()
    lrn_ref_mag = np.abs(lrn_ref_fourier)

    argmax_ref_spatial = np.zeros_like(lrn_ref_spatial)
    argmax_ref_idx = np.argmax(lrn_ref_spatial, axis=0)
    argmax_ref_spatial[argmax_ref_idx, np.arange(n)] = 1.0

    n_rows = n_rotors + 1
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col_idx, title in enumerate(
        ["Target (Spatial)", "Learned (Spatial)", "Learned (Argmax)", "Target Q (|Fourier|)", "Learned Q (|Fourier|)"]
    ):
        axes[0, col_idx].set_title(title, fontsize=11, weight="bold", pad=8)

    def draw_numbers(ax, matrix, fmt="{:.2f}", threshold=0.5):
        for r in range(n):
            for c in range(n):
                val = float(np.abs(matrix[r, c]))
                text = fmt.format(matrix[r, c].real if np.isrealobj(matrix) else val)
                colour = "white" if val > threshold else "black"
                ax.text(c, r, text, ha="center", va="center", color=colour, fontsize=8)

    def std_ticks(ax):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(list(alphabet), fontsize=8)
        ax.set_yticklabels(list(alphabet), fontsize=8)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#d3d3d3", linewidth=0.5)
        ax.grid(which="major", visible=False)

    def fourier_ticks(ax):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([str(i) for i in range(n)], fontsize=8)
        ax.set_yticklabels([str(i) for i in range(n)], fontsize=8)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#d3d3d3", linewidth=0.5)
        ax.grid(which="major", visible=False)

    for row_idx, rd in enumerate(rotor_data):
        ax0, ax1, ax2, ax3, ax4 = axes[row_idx]

        ax0.imshow(rd["tgt_spatial"], cmap="Blues", vmin=0, vmax=1)
        std_ticks(ax0)
        ax0.set_ylabel(rd["label"], fontsize=10, weight="bold")

        max_sp = max(1.0, float(np.max(np.abs(rd["lrn_spatial"]))))
        im1 = ax1.imshow(rd["lrn_spatial"], cmap="coolwarm", vmin=-max_sp, vmax=max_sp)
        std_ticks(ax1)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2.imshow(rd["argmax_spatial"], cmap="Blues", vmin=0, vmax=1)
        std_ticks(ax2)

        ax3.imshow(rd["tgt_Q_mag"], cmap="Purples", vmin=0, vmax=1)
        fourier_ticks(ax3)

        max_q = max(1.0, float(np.max(rd["lrn_Q_mag"])))
        im4 = ax4.imshow(rd["lrn_Q_mag"], cmap="Purples", vmin=0, vmax=max_q)
        fourier_ticks(ax4)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        if show_numbers:
            draw_numbers(ax0, rd["tgt_spatial"], fmt="{:.0f}", threshold=0.5)
            draw_numbers(ax1, rd["lrn_spatial"], threshold=max_sp * 0.5)
            draw_numbers(ax2, rd["argmax_spatial"], fmt="{:.0f}", threshold=0.5)
            draw_numbers(ax3, rd["tgt_Q_mag"], threshold=0.5)
            draw_numbers(ax4, rd["lrn_Q_mag"], threshold=max_q * 0.5)

    ref_row = n_rotors
    ax0, ax1, ax2, ax3, ax4 = axes[ref_row]

    ax0.imshow(tgt_ref_spatial, cmap="Blues", vmin=0, vmax=1)
    std_ticks(ax0)
    ax0.set_ylabel("Reflector", fontsize=10, weight="bold")

    max_sp = max(1.0, float(np.max(np.abs(lrn_ref_spatial))))
    im1 = ax1.imshow(lrn_ref_spatial, cmap="coolwarm", vmin=-max_sp, vmax=max_sp)
    std_ticks(ax1)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2.imshow(argmax_ref_spatial, cmap="Blues", vmin=0, vmax=1)
    std_ticks(ax2)

    ax3.imshow(tgt_ref_mag, cmap="Purples", vmin=0, vmax=1)
    fourier_ticks(ax3)

    max_q = max(1.0, float(np.max(lrn_ref_mag)))
    im4 = ax4.imshow(lrn_ref_mag, cmap="Purples", vmin=0, vmax=max_q)
    fourier_ticks(ax4)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    if show_numbers:
        draw_numbers(ax0, tgt_ref_spatial, fmt="{:.0f}", threshold=0.5)
        draw_numbers(ax1, lrn_ref_spatial, threshold=max_sp * 0.5)
        draw_numbers(ax2, argmax_ref_spatial, fmt="{:.0f}", threshold=0.5)
        draw_numbers(ax3, tgt_ref_mag, threshold=0.5)
        draw_numbers(ax4, lrn_ref_mag, threshold=max_q * 0.5)

    plt.suptitle("Q-Net — Fourier-domain rotor analysis", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
