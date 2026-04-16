import numpy as np
import matplotlib.pyplot as plt

def logisticMap(x, y, r1, r2, eps):
    x_next = (1 - eps) * r1 * x * (1 - x) + eps * r2 * y * (1 - y)
    y_next = (1 - eps) * r2 * y * (1 - y) + eps * r1 * x * (1 - x)
    return x_next, y_next

def findFixedPoints(r1, r2, eps, nIter=1000):
    x, y = 0.5, 0.5
    for _ in range(nIter):
        x, y = logisticMap(x, y, r1, r2, eps)
    return x, y

r1, r2 = 4.0, 3.1
eps_values = np.linspace(0, 0.5, 100)

x_vals = []
y_vals = []

for eps in eps_values:
    x_fp, y_fp = findFixedPoints(r1, r2, eps)
    x_vals.append(x_fp)
    y_vals.append(y_fp)

plt.plot(eps_values, x_vals, label="x*")
plt.plot(eps_values, y_vals, label="y*")
plt.xlabel("epsilon")
plt.ylabel("Fixed point value")
plt.legend()
plt.title("Fixed points vs coupling")
plt.show()