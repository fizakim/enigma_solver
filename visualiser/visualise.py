import numpy as np
import matplotlib.pyplot as plt

def visualise(net, target_sim, show_active=True):
    alphabet = net.config.alphabet
    n = len(alphabet)
    K = len(net.rotors)
    
    fig, axes = plt.subplots(2, K + 1, figsize=(3 * (K + 1), 7.5))
    
    for col in range(K):
        rotor_idx = K - 1 - col
        target_r = target_sim.rotors[rotor_idx]
        net_r = net.rotors[rotor_idx]
        
        p_t = target_r.position if show_active else 0
        p_n = net.positions[rotor_idx] if show_active else 0
        
        M_rotor = target_r.matrix
        if show_active:
            M_rotor = np.roll(np.roll(M_rotor, -p_t, axis=1), -p_t, axis=0)
            
        axes[0, col].imshow(M_rotor, cmap='Blues', vmin=0, vmax=1)
        axes[0, col].set_title(f"Rotor {rotor_idx + 1}")
        
        for r in range(n):
            for c_val in range(n):
                val = M_rotor[r, c_val]
                text_str = f"{int(round(val))}" if abs(val - round(val)) < 1e-4 else f"{val:.2f}"
                color = "white" if val > 0.5 else "black"
                axes[0, col].text(c_val, r, text_str, ha="center", va="center", color=color, fontsize=10)
        
        W_rotor = net_r.get_wiring().detach().cpu().numpy()
        if show_active:
            W_rotor = np.roll(np.roll(W_rotor, -p_n, axis=1), -p_n, axis=0)
            
        max_abs = max(1.0, float(np.max(np.abs(W_rotor))))
        im = axes[1, col].imshow(W_rotor, cmap='coolwarm', vmin=-max_abs, vmax=max_abs)
        axes[1, col].set_title(f"Rotor {rotor_idx + 1} Weights")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

        for r in range(n):
            for c_val in range(n):
                val = W_rotor[r, c_val]
                text_str = f"{val:.2f}"
                color = "white" if abs(val) > max_abs * 0.5 else "black"
                axes[1, col].text(c_val, r, text_str, ha="center", va="center", color=color, fontsize=10)

    target_ref = target_sim.reflector.matrix
    axes[0, K].imshow(target_ref, cmap='Blues', vmin=0, vmax=1)
    axes[0, K].set_title("Reflector")
    for r in range(n):
        for c_val in range(n):
            val = target_ref[r, c_val]
            text_str = f"{int(round(val))}" if abs(val - round(val)) < 1e-4 else f"{val:.2f}"
            color = "white" if val > 0.5 else "black"
            axes[0, K].text(c_val, r, text_str, ha="center", va="center", color=color, fontsize=10)
    
    net_ref = net.reflector.detach().cpu().numpy()
    max_abs_ref = max(1.0, float(np.max(np.abs(net_ref))))
    im_ref = axes[1, K].imshow(net_ref, cmap='coolwarm', vmin=-max_abs_ref, vmax=max_abs_ref)
    axes[1, K].set_title("Reflector")
    fig.colorbar(im_ref, ax=axes[1, K], fraction=0.046, pad=0.04)
    for r in range(n):
        for c_val in range(n):
            val = net_ref[r, c_val]
            text_str = f"{val:.2f}"
            color = "white" if abs(val) > max_abs_ref * 0.5 else "black"
            axes[1, K].text(c_val, r, text_str, ha="center", va="center", color=color, fontsize=10)

    for ax in axes.flat:
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='#d3d3d3', linestyle='-', linewidth=0.5)
        ax.grid(which='major', visible=False)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(list(alphabet), fontsize=8)
        ax.set_yticklabels(list(alphabet), fontsize=8)

    axes[0, 0].set_ylabel("Target", fontsize=12, weight='bold')
    axes[1, 0].set_ylabel("Net", fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.show()
