import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def fixedPointEquations(vars, r1, r2, eps):
    x, y = vars
    eq1 = x - ((1 - eps) * r1 * x * (1 - x) + eps * r2 * y * (1 - y))
    eq2 = y - ((1 - eps) * r2 * y * (1 - y) + eps * r1 * x * (1 - x))
    return [eq1, eq2]

r1, r2 = 3.4, 4.0
eps_values = np.linspace(0, 0.5, 100)

x_vals = []
y_vals = []

for eps in eps_values:
    x_fp, y_fp = fsolve(fixedPointEquations, [0.5, 0.5], args=(r1, r2, eps))
    x_vals.append(x_fp)
    y_vals.append(y_fp)

plt.plot(eps_values, x_vals, label="x*")
plt.plot(eps_values, y_vals, label="y*")
plt.xlabel("epsilon")
plt.ylabel("Fixed point value")
plt.legend()
plt.title("Fixed points vs coupling")
plt.show()
