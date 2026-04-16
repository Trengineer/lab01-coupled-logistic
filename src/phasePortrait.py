"""
Exercise 3: Phase portraits of the coupled logistic map.

Iterates the map and plots orbits in the (x, y) phase plane.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('./figures', exist_ok=True)


# --- Parameters ---
PARAM_SETS = [
    (2.8, 2.9, 0.1),   # expect convergence to fixed point
    (3.1, 3.4, 0.3),   # periodic regime
    (3.85, 3.95, 0.2),  # chaotic regime
]

INITIAL_CONDITION = (0.2, 0.3)
N_TRANSIENT = 500
N_KEEP = 10000


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



if __name__ == "__main__":
    x0, y0 = INITIAL_CONDITION
    nSteps = N_TRANSIENT + N_KEEP

    fig, axes = plt.subplots(1, len(PARAM_SETS), figsize=(5 * len(PARAM_SETS), 5))

    for ax, (r1, r2, eps) in zip(axes, PARAM_SETS):
        x, y = solveOrbit(nSteps, x0, y0, r1, r2, eps)
        x_keep = x[N_TRANSIENT:]
        y_keep = y[N_TRANSIENT:]

        ax.scatter(x_keep, y_keep, s=1)
        ax.set_xlabel(r'$x_n$')
        ax.set_ylabel(r'$y_n$')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(fr'$r_1={r1},\; r_2={r2},\; \varepsilon={eps}$')

    fig.tight_layout()
    plt.savefig(f'./figures/phaseportrait.png')
    plt.show()
