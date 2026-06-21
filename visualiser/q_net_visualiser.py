import numpy as np
import matplotlib.pyplot as plt
import torch

def get_dft_matrix(n):
    # Construct unitary DFT matrix F of size n x n
    # F[k, m] = 1/sqrt(n) * exp(-2j * pi * k * m / n)
    k = np.arange(n).reshape((n, 1))
    m = np.arange(n).reshape((1, n))
    F = np.exp(-2j * np.pi * k * m / n) / np.sqrt(n)
    return F

def visualise_q_net(net, target_sim, position=None, show_numbers=True):
    alphabet = net.config.alphabet
    n = len(alphabet)
    
    if position is None:
        position = [0] * len(net.config.rotors)
        
    # Reset and step both net and target_sim
    net.reset(position)
    net.step()
    
    target_sim.reset(position)
    for r in reversed(target_sim.rotors):
        if not r.step():
            break
            
    # Get the active positions
    active_pos_net = net.positions
    
    # 1. Compute target Q (standard basis)
    Q_target = torch.eye(n)
    for r in target_sim.rotors:
        r_matrix = torch.from_numpy(r.matrix).float()
        r_pos = r.position
        W_eff = torch.roll(r_matrix, shifts=(-r_pos, -r_pos), dims=(0, 1))
        Q_target = Q_target @ W_eff
    Q_target = Q_target.numpy()
    
    # 2. Get QNet Q (standard basis)
    with torch.no_grad():
        Q_net = net.get_Q(active_pos_net).detach().cpu().numpy()
        
    # 3. Compute Fourier basis representation of Q_target and Q_net
    F = get_dft_matrix(n)
    Q_target_fourier = F @ Q_target @ F.conj().T
    Q_net_fourier = F @ Q_net @ F.conj().T
    
    Q_target_fourier_mag = np.abs(Q_target_fourier)
    Q_net_fourier_mag = np.abs(Q_net_fourier)
    
    # Get reflectors
    target_ref = target_sim.reflector.matrix
    with torch.no_grad():
        net_ref = net.reflector.detach().cpu().numpy()
        
    # Compute Fourier basis representation of reflectors
    target_ref_fourier = F @ target_ref @ F.conj().T
    net_ref_fourier = F @ net_ref @ F.conj().T
    
    target_ref_fourier_mag = np.abs(target_ref_fourier)
    net_ref_fourier_mag = np.abs(net_ref_fourier)
    
    # Plotting setup: 4 rows, 2 columns
    fig, axes = plt.subplots(4, 2, figsize=(8.5, 15))
    
    # Helper to draw numbers on imshow plot
    def draw_numbers(ax, matrix, fmt="{:.2f}", is_standard=True, max_val=1.0):
        for r in range(n):
            for c_val in range(n):
                val = matrix[r, c_val]
                if is_standard and abs(val - round(val)) < 1e-4:
                    text_str = f"{int(round(val))}"
                else:
                    text_str = fmt.format(val)
                
                if is_standard:
                    color = "white" if val > 0.5 * max_val else "black"
                else:
                    color = "white" if abs(val) > 0.5 * max_val else "black"
                ax.text(c_val, r, text_str, ha="center", va="center", color=color, fontsize=10)

    # ROW 1: Target (Standard basis)
    # Q
    axes[0, 0].imshow(Q_target, cmap='Blues', vmin=0, vmax=1)
    axes[0, 0].set_title("Target Q (Standard Basis)")
    # Reflector
    axes[0, 1].imshow(target_ref, cmap='Blues', vmin=0, vmax=1)
    axes[0, 1].set_title("Target Reflector (Standard Basis)")
    
    # ROW 2: Net normal basis (Standard basis)
    # Q
    max_abs_q = max(1.0, float(np.max(np.abs(Q_net))))
    im_q_net = axes[1, 0].imshow(Q_net, cmap='coolwarm', vmin=-max_abs_q, vmax=max_abs_q)
    axes[1, 0].set_title("QNet Q (Standard Basis)")
    fig.colorbar(im_q_net, ax=axes[1, 0], fraction=0.046, pad=0.04)
    # Reflector
    max_abs_ref = max(1.0, float(np.max(np.abs(net_ref))))
    im_ref = axes[1, 1].imshow(net_ref, cmap='coolwarm', vmin=-max_abs_ref, vmax=max_abs_ref)
    axes[1, 1].set_title("QNet Reflector (Standard Basis)")
    fig.colorbar(im_ref, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # ROW 3: Target Fourier Basis
    # Q
    axes[2, 0].imshow(Q_target_fourier_mag, cmap='Purples', vmin=0, vmax=1)
    axes[2, 0].set_title("Target Q (Fourier Basis Mag)")
    # Reflector
    axes[2, 1].imshow(target_ref_fourier_mag, cmap='Purples', vmin=0, vmax=1)
    axes[2, 1].set_title("Target Reflector (Fourier Basis Mag)")
    
    # ROW 4: Net Fourier Basis
    # Q
    max_abs_q_f = max(1.0, float(np.max(np.abs(Q_net_fourier_mag))))
    im_q_net_f = axes[3, 0].imshow(Q_net_fourier_mag, cmap='Purples', vmin=0, vmax=max_abs_q_f)
    axes[3, 0].set_title("QNet Q (Fourier Basis Mag)")
    fig.colorbar(im_q_net_f, ax=axes[3, 0], fraction=0.046, pad=0.04)
    # Reflector
    max_abs_ref_f = max(1.0, float(np.max(np.abs(net_ref_fourier_mag))))
    im_ref_f = axes[3, 1].imshow(net_ref_fourier_mag, cmap='Purples', vmin=0, vmax=max_abs_ref_f)
    axes[3, 1].set_title("QNet Reflector (Fourier Basis Mag)")
    fig.colorbar(im_ref_f, ax=axes[3, 1], fraction=0.046, pad=0.04)
    
    if show_numbers:
        draw_numbers(axes[0, 0], Q_target, is_standard=True)
        draw_numbers(axes[0, 1], target_ref, is_standard=True)
        
        draw_numbers(axes[1, 0], Q_net, fmt="{:.2f}", is_standard=False, max_val=max_abs_q)
        draw_numbers(axes[1, 1], net_ref, fmt="{:.2f}", is_standard=False, max_val=max_abs_ref)
        
        draw_numbers(axes[2, 0], Q_target_fourier_mag, fmt="{:.2f}", is_standard=False)
        draw_numbers(axes[2, 1], target_ref_fourier_mag, fmt="{:.2f}", is_standard=False)
        
        draw_numbers(axes[3, 0], Q_net_fourier_mag, fmt="{:.2f}", is_standard=False, max_val=max_abs_q_f)
        draw_numbers(axes[3, 1], net_ref_fourier_mag, fmt="{:.2f}", is_standard=False, max_val=max_abs_ref_f)
        
    # Formatting ticks and grids
    # Standard basis ticks (alphabet)
    for r_idx in [0, 1]:
        for c_idx in [0, 1]:
            ax = axes[r_idx, c_idx]
            ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
            ax.grid(which='minor', color='#d3d3d3', linestyle='-', linewidth=0.5)
            ax.grid(which='major', visible=False)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(list(alphabet), fontsize=8)
            ax.set_yticklabels(list(alphabet), fontsize=8)
            
    # Fourier basis ticks (frequency indices)
    for r_idx in [2, 3]:
        for c_idx in [0, 1]:
            ax = axes[r_idx, c_idx]
            ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
            ax.grid(which='minor', color='#d3d3d3', linestyle='-', linewidth=0.5)
            ax.grid(which='major', visible=False)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([str(i) for i in range(n)], fontsize=8)
            ax.set_yticklabels([str(i) for i in range(n)], fontsize=8)
            
    axes[0, 0].set_ylabel("Target (Std)", fontsize=12, weight='bold')
    axes[1, 0].set_ylabel("QNet (Std)", fontsize=12, weight='bold')
    axes[2, 0].set_ylabel("Target (Fourier)", fontsize=12, weight='bold')
    axes[3, 0].set_ylabel("QNet (Fourier)", fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.show()
