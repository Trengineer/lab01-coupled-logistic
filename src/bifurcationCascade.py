"""
Exercise 4 — Bifurcation cascade.

"""

import numpy as np
import matplotlib.pyplot as plt


def coupled_map(x, y, r1, r2, eps):
    """One step of the coupled logistic map."""
    f_x = r1 * x * (1.0 - x)      
    f_y = r2 * y * (1.0 - y)      
    x_new = (1.0 - eps) * f_x + eps * f_y
    y_new = (1.0 - eps) * f_y + eps * f_x
    return x_new, y_new


R1       = 3.1          # fixed growth rate for species 1
EPS      = 0.3          # coupling strength
R2_MIN   = 2.8          # start of sweep
R2_MAX   = 3.97         # end of sweep
N_R2     = 800          # number of r2 values in the sweep

N_TRANSIENT = 500       # iterations thrown away before recording
N_RECORD    = 300       # iterations kept after the transient

X0, Y0   = 0.2, 0.3    # initial condition (arbitrary, inside [0,1])


def compute_bifurcation_data(r1, eps, r2_values, x0, y0,
                              n_transient, n_record):
    
    total = len(r2_values) * n_record
    r2_col = np.empty(total)
    x_col  = np.empty(total)

    for i, r2 in enumerate(r2_values):
        x, y = x0, y0

        # ── Transient: iterate and throw away ──
        for _ in range(n_transient):
            x, y = coupled_map(x, y, r1, r2, eps)

        # ── Record phase: store x_n values ──
        start = i * n_record
        for j in range(n_record):
            x, y = coupled_map(x, y, r1, r2, eps)
            r2_col[start + j] = r2
            x_col[start + j]  = x

    return r2_col, x_col



def main():
    r2_values = np.linspace(R2_MIN, R2_MAX, N_R2)

    r2_col, x_col = compute_bifurcation_data(
        R1, EPS, r2_values, X0, Y0, N_TRANSIENT, N_RECORD
    )

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(r2_col, x_col, ',', color='black', markersize=0.2, alpha=0.5)

    ax.set_xlabel(r'$r_2$', fontsize=14)
    ax.set_ylabel(r'$x_n$ (long-time values)', fontsize=14)
    ax.set_title(
        rf'Bifurcation cascade — $r_1 = {R1}$, $\varepsilon = {EPS}$',
        fontsize=15,
    )
    ax.set_xlim(R2_MIN, R2_MAX)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig('../figures/bifurcation_cascade.png', dpi=250)
    plt.show()
    print('Saved to ../figures/bifurcation_cascade.png')


if __name__ == '__main__':
    main()