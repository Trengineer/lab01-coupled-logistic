"""
Exercise 4: Bifurcation cascade for the coupled logistic map.

Fix r1 = 3.1, eps = 0.3, sweep r2 in [2.8, 3.97].
Plot long-time values of x_n vs r2.
"""

import numpy as np
import matplotlib.pyplot as plt
from coupled_map import iterate

# --- Parameters ---
R1 = 3.1
EPS = 0.3
R2_MIN = 2.8
R2_MAX = 3.97
N_R2 = 800

INITIAL_CONDITION = (0.2, 0.3)
N_TRANSIENT = 500
N_KEEP = 300


def main():
    r2_values = np.linspace(R2_MIN, R2_MAX, N_R2)

    # Pre-allocate arrays for the scatter plot
    all_r2 = np.empty(N_R2 * N_KEEP)
    all_x = np.empty(N_R2 * N_KEEP)

    for i, r2 in enumerate(r2_values):
        xs, _ = iterate(
            *INITIAL_CONDITION, R1, r2, EPS,
            n_transient=N_TRANSIENT, n_keep=N_KEEP,
        )
        all_r2[i * N_KEEP : (i + 1) * N_KEEP] = r2
        all_x[i * N_KEEP : (i + 1) * N_KEEP] = xs

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(all_r2, all_x, ",", color="black", markersize=0.2)
    ax.set_xlabel("$r_2$")
    ax.set_ylabel("$x_n$ (long-time values)")
    ax.set_title(f"Bifurcation cascade — $r_1 = {R1}$, $\\varepsilon = {EPS}$")
    ax.set_xlim(R2_MIN, R2_MAX)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig("../figures/bifurcation_cascade.png", dpi=200)
    plt.show()
    print("Saved to ../figures/bifurcation_cascade.png")


if __name__ == "__main__":
    main()
