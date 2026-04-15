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
N_TRANSIENT = 10
N_KEEP = 1000


def f(x, y, r1, r2, eps):
    """Return one step of the logistic map."""
    x_n = (1 - eps)*r1*x*(1-x) + eps*r2*y*(1-y)
    y_n = (1 - eps)*r2*y*(1-y) + eps*r1*x*(1-x)
    
    return x_n, y_n

def solveOrbit(nSteps, x0, y0, r1, r2, eps):
    """Compute the forward orbit starting from x0 and y0."""
    x = np.zeros(nSteps + 1)
    y = np.zeros(nSteps + 1)
    x[0] = x0
    y[0] = y0
    
    for i in range(nSteps):
        x[i + 1], y[i + 1] = f(x[i],y[i], r1, r2, eps)
    return x, y



# def plot_phase_portrait(r1, r2, eps, ax):
#     """Plot a single phase portrait on the given axes."""
#     xs, ys = iterate(
#         *INITIAL_CONDITION, r1, r2, eps,
#         n_transient=N_TRANSIENT, n_keep=N_KEEP,
#     )
#     ax.plot(xs, ys, ",", color="C0", markersize=0.5)
#     ax.set_xlabel("$x_n$")
#     ax.set_ylabel("$y_n$")
#     ax.set_title(f"$r_1={r1},\\; r_2={r2},\\; \\varepsilon={eps}$")
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_aspect("equal")


# def main():
#     fig, axes = plt.subplots(1, len(PARAM_SETS), figsize=(5 * len(PARAM_SETS), 5))
#     if len(PARAM_SETS) == 1:
#         axes = [axes]

#     for ax, (r1, r2, eps) in zip(axes, PARAM_SETS):
#         plot_phase_portrait(r1, r2, eps, ax)

#     fig.tight_layout()
#     fig.savefig("../figures/phase_portraits.png", dpi=200)
#     plt.show()
#     print("Saved to ../figures/phase_portraits.png")


if __name__ == "__main__":
    x0, y0 = INITIAL_CONDITION
    r1, r2, eps = PARAM_SETS[2]
    nSteps = N_TRANSIENT + N_KEEP 
    x, y = solveOrbit(nSteps, x0, y0, r1, r2, eps)

    x_keep = x[N_TRANSIENT:]
    y_keep = y[N_TRANSIENT:]

    plt.scatter(x_keep, y_keep)
    plt.xlabel(fr'$x_n$')
    plt.ylabel(fr'$y_n$')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
