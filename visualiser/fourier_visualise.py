import numpy as np
import matplotlib.pyplot as plt
import torch

def visualise_fourier(net, target_sim, show_active=True, show_numbers=True):
    alphabet = net.config.alphabet
    n = len(alphabet)
    K = len(net.rotors)
    
    fig, axes = plt.subplots(3, K + 1, figsize=(3.5 * (K + 1), 11))
    
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
        
        if show_numbers:
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
        
        if show_numbers:
            for r in range(n):
                for c_val in range(n):
                    val = W_rotor[r, c_val]
                    text_str = f"{val:.2f}"
                    color = "white" if abs(val) > max_abs * 0.5 else "black"
                    axes[1, col].text(c_val, r, text_str, ha="center", va="center", color=color, fontsize=10)
                    
        logits = net_r.logits.detach().cpu()
        half = n // 2
        real = logits[:half]
        imag = logits[half:]
        if n % 2 == 0:
            imag = torch.cat([imag, torch.zeros(1, device=logits.device)])
        mags = torch.sqrt(real**2 + imag**2).numpy()
        
        axes[2, col].bar(range(1, half + 1), mags, color='purple', alpha=0.7)
        axes[2, col].set_title(f"Rotor {rotor_idx + 1} Logits Magnitude")
        axes[2, col].set_xlabel("Frequency (k)")
        axes[2, col].set_ylabel("Magnitude")
        axes[2, col].set_xticks(range(1, half + 1))
        axes[2, col].set_ylim(bottom=0)

    target_ref = target_sim.reflector.matrix
    axes[0, K].imshow(target_ref, cmap='Blues', vmin=0, vmax=1)
    axes[0, K].set_title("Reflector")
    if show_numbers:
        for r in range(n):
            for c_val in range(n):
                val = target_ref[r, c_val]
                text_str = f"{int(round(val))}" if abs(val - round(val)) < 1e-4 else f"{val:.2f}"
                color = "white" if val > 0.5 else "black"
                axes[0, K].text(c_val, r, text_str, ha="center", va="center", color=color, fontsize=10)
    
    net_ref = net.reflector.detach().cpu().numpy()
    max_abs_ref = max(1.0, float(np.max(np.abs(net_ref))))
    im_ref = axes[1, K].imshow(net_ref, cmap='coolwarm', vmin=-max_abs_ref, vmax=max_abs_ref)
    axes[1, K].set_title("Reflector Weights")
    fig.colorbar(im_ref, ax=axes[1, K], fraction=0.046, pad=0.04)
    if show_numbers:
        for r in range(n):
            for c_val in range(n):
                val = net_ref[r, c_val]
                text_str = f"{val:.2f}"
                color = "white" if abs(val) > max_abs_ref * 0.5 else "black"
                axes[1, K].text(c_val, r, text_str, ha="center", va="center", color=color, fontsize=10)
                
    ref_logits = net.reflector_layer.logits.detach().cpu()
    half = n // 2
    real = ref_logits[:half]
    imag = ref_logits[half:]
    if n % 2 == 0:
        imag = torch.cat([imag, torch.zeros(1, device=ref_logits.device)])
    ref_mags = torch.sqrt(real**2 + imag**2).numpy()
    
    axes[2, K].bar(range(1, half + 1), ref_mags, color='purple', alpha=0.7)
    axes[2, K].set_title("Reflector Logits Magnitude")
    axes[2, K].set_xlabel("Frequency (k)")
    axes[2, K].set_ylabel("Magnitude")
    axes[2, K].set_xticks(range(1, half + 1))
    axes[2, K].set_ylim(bottom=0)

    for r_idx in [0, 1]:
        for c_idx in range(K + 1):
            ax = axes[r_idx, c_idx]
            ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
            ax.grid(which='minor', color='#d3d3d3', linestyle='-', linewidth=0.5)
            ax.grid(which='major', visible=False)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(list(alphabet), fontsize=8)
            ax.set_yticklabels(list(alphabet), fontsize=8)

    axes[0, 0].set_ylabel("Target", fontsize=12, weight='bold')
    axes[1, 0].set_ylabel("Net (Spatial)", fontsize=12, weight='bold')
    axes[2, 0].set_ylabel("Net (Spectral)", fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.show()
