"""
Exercise 3: Phase portraits of the coupled logistic map.

Iterates the map and plots orbits in the (x, y) phase plane.
"""

import numpy as np
import matplotlib.pyplot as plt
from coupled_map import coupled_map, iterate

# --- Parameters ---
PARAM_SETS = [
    (2.8, 2.9, 0.1),   # expect convergence to fixed point
    (3.1, 3.4, 0.3),   # periodic regime
    (3.85, 3.95, 0.2),  # chaotic regime
]

INITIAL_CONDITION = (0.2, 0.3)
N_TRANSIENT = 1000
N_KEEP = 5000


def plot_phase_portrait(r1, r2, eps, ax):
    """Plot a single phase portrait on the given axes."""
    xs, ys = iterate(
        *INITIAL_CONDITION, r1, r2, eps,
        n_transient=N_TRANSIENT, n_keep=N_KEEP,
    )
    ax.plot(xs, ys, ",", color="C0", markersize=0.5)
    ax.set_xlabel("$x_n$")
    ax.set_ylabel("$y_n$")
    ax.set_title(f"$r_1={r1},\\; r_2={r2},\\; \\varepsilon={eps}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def main():
    fig, axes = plt.subplots(1, len(PARAM_SETS), figsize=(5 * len(PARAM_SETS), 5))
    if len(PARAM_SETS) == 1:
        axes = [axes]

    for ax, (r1, r2, eps) in zip(axes, PARAM_SETS):
        plot_phase_portrait(r1, r2, eps, ax)

    fig.tight_layout()
    fig.savefig("../figures/phase_portraits.png", dpi=200)
    plt.show()
    print("Saved to ../figures/phase_portraits.png")


if __name__ == "__main__":
    main()
