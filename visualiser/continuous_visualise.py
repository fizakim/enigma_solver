import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import torch


def _dft_matrix(n):
    k = np.arange(n).reshape(n, 1)
    m = np.arange(n).reshape(1, n)
    return np.exp(-2j * np.pi * k * m / n) / np.sqrt(n)


def _best_candidate(net):
    """Return 0 — positions are fixed integers so any candidate is equally integer-like."""
    return 0


# ---------------------------------------------------------------------------
# Position-landscape section
# ---------------------------------------------------------------------------

def _draw_position_section(axes_row, net, true_positions):
    """Fill the pre-created axes (one per rotor) with fixed-position histograms.

    Each subplot shows how many candidates start at each integer position.
    The selected (best) candidate's position is highlighted in blue; the true
    initial position is marked with a gold dashed line.
    """
    n = net.n
    positions = net.initial_positions.cpu()  # [C, num_rotors], integer
    C = positions.shape[0]
    alphabet = net.config.alphabet
    best_c = _best_candidate(net)

    for r_idx, ax in enumerate(axes_row):
        pos_r = positions[:, r_idx].numpy()   # [C] integer positions

        # Count candidates per position
        counts = np.bincount(pos_r, minlength=n)
        bar_colors = ["steelblue"] * n

        # Highlight the best candidate's position
        best_pos = int(pos_r[best_c])
        bar_colors[best_pos] = "deepskyblue"

        ax.bar(range(n), counts, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)

        # True position marker
        if true_positions is not None:
            tp = true_positions[r_idx]
            ax.axvline(tp, color="gold", linewidth=2.5, linestyle="--", zorder=4,
                       label=f"true={alphabet[tp]}")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

        ax.set_xlim(-0.7, n - 0.3)
        ax.set_xticks(range(n))
        ax.set_xticklabels(list(alphabet), fontsize=9)
        ax.set_ylabel("# candidates", fontsize=8)
        ax.set_title(f"Rotor {r_idx}  initial positions", fontsize=10, weight="bold")
        ax.set_xlabel("position", fontsize=8)


# ---------------------------------------------------------------------------
# Wiring-analysis section  (mirrors q_net_visualiser layout)
# ---------------------------------------------------------------------------

def _draw_wiring_section(axes_grid, net, target_sim, best_c, show_numbers):
    """Fill a (num_rotors+1) × 5 axes grid with Q-matrix analysis for candidate best_c.

    Columns: Target (spatial) | Learned (spatial) | Argmax | Target |Q| | Learned |Q|
    Rows: one per rotor, then the reflector.
    """
    n = net.n
    alphabet = net.config.alphabet
    F_np = _dft_matrix(n)
    F_inv_np = F_np.conj().T

    # ---- helpers ----
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

    def freq_ticks(ax):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([str(i) for i in range(n)], fontsize=8)
        ax.set_yticklabels([str(i) for i in range(n)], fontsize=8)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#d3d3d3", linewidth=0.5)
        ax.grid(which="major", visible=False)

    # Column headers on first row only
    col_titles = [
        "Target (spatial)", "Learned (spatial)", "Argmax",
        "Target |Q|", f"Learned |Q|  (c={best_c})",
    ]
    for col_idx, title in enumerate(col_titles):
        axes_grid[0][col_idx].set_title(title, fontsize=10, weight="bold", pad=6)

    # ---- rotors ----
    for row_idx, (net_rotor, tgt_rotor) in enumerate(
        zip(net.rotors, target_sim.rotors)
    ):
        pos = tgt_rotor.position
        W = torch.from_numpy(tgt_rotor.matrix).float()
        tgt_spatial = torch.roll(W, shifts=[-pos, -pos], dims=[0, 1]).numpy()

        tgt_Q = F_np @ tgt_spatial @ F_inv_np
        tgt_Q_mag = np.abs(tgt_Q)

        with torch.no_grad():
            Q_c = torch.complex(net_rotor.Q_real[best_c], net_rotor.Q_imag[best_c])
            lrn_spatial = (net.F_inv @ Q_c @ net.F).real.cpu().numpy()
            lrn_Q_mag = np.abs(Q_c.cpu().numpy())

        argmax_spatial = np.zeros_like(lrn_spatial)
        argmax_spatial[np.argmax(lrn_spatial, axis=0), np.arange(n)] = 1.0

        ax0, ax1, ax2, ax3, ax4 = axes_grid[row_idx]

        ax0.imshow(tgt_spatial, cmap="Blues", vmin=0, vmax=1)
        std_ticks(ax0)
        ax0.set_ylabel(f"Rotor {row_idx}  (pos={pos})", fontsize=9, weight="bold")

        max_sp = max(1.0, float(np.max(np.abs(lrn_spatial))))
        im1 = ax1.imshow(lrn_spatial, cmap="coolwarm", vmin=-max_sp, vmax=max_sp)
        std_ticks(ax1)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2.imshow(argmax_spatial, cmap="Blues", vmin=0, vmax=1)
        std_ticks(ax2)

        ax3.imshow(tgt_Q_mag, cmap="Purples", vmin=0, vmax=1)
        freq_ticks(ax3)

        max_q = max(1.0, float(np.max(lrn_Q_mag)))
        im4 = ax4.imshow(lrn_Q_mag, cmap="Purples", vmin=0, vmax=max_q)
        freq_ticks(ax4)
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        if show_numbers:
            draw_numbers(ax0, tgt_spatial, fmt="{:.0f}", threshold=0.5)
            draw_numbers(ax1, lrn_spatial, threshold=max_sp * 0.5)
            draw_numbers(ax2, argmax_spatial, fmt="{:.0f}", threshold=0.5)
            draw_numbers(ax3, tgt_Q_mag, threshold=0.5)
            draw_numbers(ax4, lrn_Q_mag, threshold=max_q * 0.5)

    # ---- reflector ----
    ref_row = len(net.rotors)
    ax0, ax1, ax2, ax3, ax4 = axes_grid[ref_row]

    tgt_ref = target_sim.reflector.matrix
    tgt_ref_Q = F_np @ tgt_ref @ F_inv_np
    tgt_ref_mag = np.abs(tgt_ref_Q)

    with torch.no_grad():
        lrn_ref = net.reflector.cpu().numpy()
        lrn_ref_Q = net.reflector_fourier.cpu().numpy()
    lrn_ref_mag = np.abs(lrn_ref_Q)

    argmax_ref = np.zeros_like(lrn_ref)
    argmax_ref[np.argmax(lrn_ref, axis=0), np.arange(n)] = 1.0

    ax0.imshow(tgt_ref, cmap="Blues", vmin=0, vmax=1)
    std_ticks(ax0)
    ax0.set_ylabel("Reflector", fontsize=9, weight="bold")

    max_sp = max(1.0, float(np.max(np.abs(lrn_ref))))
    im1 = ax1.imshow(lrn_ref, cmap="coolwarm", vmin=-max_sp, vmax=max_sp)
    std_ticks(ax1)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2.imshow(argmax_ref, cmap="Blues", vmin=0, vmax=1)
    std_ticks(ax2)

    ax3.imshow(tgt_ref_mag, cmap="Purples", vmin=0, vmax=1)
    freq_ticks(ax3)

    max_q = max(1.0, float(np.max(lrn_ref_mag)))
    im4 = ax4.imshow(lrn_ref_mag, cmap="Purples", vmin=0, vmax=max_q)
    freq_ticks(ax4)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    if show_numbers:
        draw_numbers(ax0, tgt_ref, fmt="{:.0f}", threshold=0.5)
        draw_numbers(ax1, lrn_ref, threshold=max_sp * 0.5)
        draw_numbers(ax2, argmax_ref, fmt="{:.0f}", threshold=0.5)
        draw_numbers(ax3, tgt_ref_mag, threshold=0.5)
        draw_numbers(ax4, lrn_ref_mag, threshold=max_q * 0.5)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def visualise_continuous(
    net,
    target_sim,
    true_positions=None,
    best_candidate_idx=None,
    show_numbers=True,
):
    """Two-section visualisation for ContinuousQNet.

    Top section — candidate positions:
        One bar chart per rotor showing how many candidates start at each
        integer position.  A gold dashed line marks the true initial position
        when *true_positions* is supplied.

    Bottom section — wiring quality:
        Five-column Q-matrix analysis for a single candidate (candidate 0 by
        default, or *best_candidate_idx* if given).
        Columns: target spatial | learned spatial | argmax | target |Q| | learned |Q|.

    Parameters
    ----------
    net : ContinuousQNet
    target_sim : Enigma simulator (built from config.build())
    true_positions : list[int] | None
        True initial positions used during training.  When None the gold
        reference line is omitted from the position chart.
    best_candidate_idx : int | None
        Which candidate to use for the wiring section.  Defaults to 0.
    show_numbers : bool
        Annotate matrix cells with their values.
    """
    num_rotors = net.num_rotors
    num_wiring_rows = num_rotors + 1  # rotors + reflector

    if best_candidate_idx is None:
        best_candidate_idx = _best_candidate(net)

    # Step target_sim to match one encryption step from true_positions (or pos 0)
    step_from = true_positions if true_positions is not None else [0] * num_rotors
    target_sim.reset(step_from)
    for r in reversed(target_sim.rotors):
        if not r.step():
            break

    # ---- figure layout ----
    # Two vertically stacked sections with different column counts are handled
    # via nested GridSpec: the outer grid defines vertical slices, the inner
    # grids define columns within each slice.
    pos_height = 2.8          # inches per position-landscape row (just 1 row)
    wiring_height = 3.5       # inches per wiring row
    total_height = pos_height + wiring_height * num_wiring_rows

    fig = plt.figure(figsize=(5 * 5, total_height))
    fig.suptitle(
        f"ContinuousQNet — candidate positions & wiring quality"
        f"  (candidate {best_candidate_idx} / {net.num_candidates})",
        fontsize=13, y=1.01,
    )

    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[pos_height, wiring_height * num_wiring_rows],
        hspace=0.45,
    )

    # Top: position landscape  (1 row × num_rotors cols)
    top_gs = GridSpecFromSubplotSpec(
        1, num_rotors,
        subplot_spec=outer[0],
        wspace=0.35,
    )
    pos_axes = [fig.add_subplot(top_gs[0, r]) for r in range(num_rotors)]

    # Bottom: wiring analysis  (num_wiring_rows × 5 cols)
    bot_gs = GridSpecFromSubplotSpec(
        num_wiring_rows, 5,
        subplot_spec=outer[1],
        hspace=0.55, wspace=0.35,
    )
    wiring_axes = [
        [fig.add_subplot(bot_gs[row, col]) for col in range(5)]
        for row in range(num_wiring_rows)
    ]

    _draw_position_section(pos_axes, net, true_positions)
    _draw_wiring_section(wiring_axes, net, target_sim, best_candidate_idx, show_numbers)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    plt.show()
