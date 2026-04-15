"""
Coupled logistic map — shared definitions.

Model:
    x_{n+1} = (1 - eps) * r1 * x_n * (1 - x_n) + eps * r2 * y_n * (1 - y_n)
    y_{n+1} = (1 - eps) * r2 * y_n * (1 - y_n) + eps * r1 * x_n * (1 - x_n)
"""

import numpy as np


def coupled_map(x, y, r1, r2, eps):
    """One iteration of the coupled logistic map.

    Parameters
    ----------
    x, y : float or ndarray
        Current population densities.
    r1, r2 : float
        Growth parameters.
    eps : float
        Coupling strength in [0, 1].

    Returns
    -------
    x_new, y_new : same type as input
    """
    logistic_x = r1 * x * (1 - x)
    logistic_y = r2 * y * (1 - y)
    x_new = (1 - eps) * logistic_x + eps * logistic_y
    y_new = (1 - eps) * logistic_y + eps * logistic_x
    return x_new, y_new


def iterate(x0, y0, r1, r2, eps, n_transient=500, n_keep=300):
    """Iterate the map, discard transient, return kept iterates.

    Returns
    -------
    xs, ys : ndarray of shape (n_keep,)
    """
    x, y = x0, y0
    for _ in range(n_transient):
        x, y = coupled_map(x, y, r1, r2, eps)
    xs = np.empty(n_keep)
    ys = np.empty(n_keep)
    for i in range(n_keep):
        x, y = coupled_map(x, y, r1, r2, eps)
        xs[i] = x
        ys[i] = y
    return xs, ys


def compose(x, y, r1, r2, eps, p):
    """Apply the coupled map p times: F^p(x, y)."""
    for _ in range(p):
        x, y = coupled_map(x, y, r1, r2, eps)
    return x, y
