import numpy as np
def F(x, y, r1, r2, eps):
    """
    This is the Map function, it takes in normalized populations x and y, 
    the the intrinsic growth parameters r1 and r2 
    and lastly the coupling parameter eps (=epsilon)
    it then returns the next step/generation x1 and y1 as two floats
    
    parameters:
    :param x: normalized population at generation n
    :param y: another normalized population at generation n
    :param r1: intrinsic growth parameter for population x
    :param r2: intrinsic growth parameter for population y
    :param eps: coupling strength, in [0,1]

    returns:
    x1: normalized population at generation n + 1
    y1: another normalized population at generation n + 1

    """
    x1 = (1 - eps) * r1 * x * (1 - x)  +  eps * r2 * y * (1 - y)
    y1 = (1 - eps) * r2 * y * (1 - y) + eps * r1 * x *(1 - x)

    return x1, y1

def F2(x, y, r1, r2, eps):
    """
    same parameters as F

    returns:
    F applied two times, which is 
    x2: normalized population at generation n + 2
    y2: another normalized population at generation n + 2
    """
    x1, y1 = F(x, y, r1, r2, eps)
    return F(x1, y1, r1, r2, eps)

def F3(x, y, r1, r2, eps):
    """
    same parameters as F

    returns:
    F applied three times, which is 
    x3: normalized population at generation n + 3
    y3: another normalized population at generation n + 3
    """
    x1, y1 = F2(x, y, r1, r2, eps)
    return F(x1, y1, r1, r2, eps)

def JacobianF(x, y, r1, r2, eps):
    """
    Calculates and returns the 2x2 Jabobian matrix for F.
    The jacobian of F is defined as follows:
    we have F(x,y) = (x1, y1)
    J_f = [dx1/dx dx1/dy
           dy1/dx dy1/dy] 
    Parameters: 
    Same as in F(x, y, r1, r2, eps)
    Returns:
    The Jacobian of F as a 2x2 matrix consisting of 
    all partial derivatives of F
    """
    J_f = np.zeros((2, 2))
    J_f[0,0] = (1 - eps) * r1 -  2 * (1 - eps) * r1 * x
    J_f[0,1] =  eps * r2 - 2 *  eps * r2 * y 
    J_f[1,0] = eps * r1 - 2 * eps * r1 * x
    J_f[1,1] = (1 - eps) * r2 - 2 * (1 - eps) * r2 * y
    return J_f

def JacobianF2(x, y, r1, r2, eps):
    """
    Calculates and returns the 2x2 Jabobian matrix for F2.
    The chain rule for matrix derivatives says the Jacobian of a 
    composition is the product of the Jacobians at each point along the orbit

    Parameters: 
    Same as in F(x, y, r1, r2, eps)
    Returns:
    The Jacobian of F2 as a 2x2 matrix consisting of 
    all partial derivatives of F2
    """
    x1, y1 = F(x, y, r1, r2, eps)
    return JacobianF(x1, y1, r1, r2, eps) @ JacobianF(x, y, r1, r2, eps)

def JacobianF3(x, y, r1, r2, eps):
    """
    Calculates and returns the 2x2 Jabobian matrix for F3.
    The chain rule for matrix derivatives says the Jacobian of a 
    composition is the product of the Jacobians at each point along the orbit

    Parameters: 
    Same as in F(x, y, r1, r2, eps)
    Returns:
    The Jacobian of F3 as a 2x2 matrix consisting of 
    all partial derivatives of F3
    """
    x1, y1 = F2(x, y, r1, r2, eps)
    return JacobianF(x1, y1, r1, r2, eps) @ JacobianF2(x, y, r1, r2, eps)

def newton(x0, y0, p, r1, r2, eps, tol=1e-10, max_iter=50):
    """
    Applies Newtons method to find a period-p point of the coupled map,
    starting from the initial guess (x0, y0)
    
    :param x0: initial guess of x
    :param y0: initial guess of y
    :param p: period
    :param r1: same as before
    :param r2: same as before
    :param eps: same as before
    :param tol: tolerating factor for the newton method
    :param max_iter: max # of iterations
    Returns:
    the period-p point as (x,y) coordinates

    """
    for n in range(max_iter):
        if p == 2:
            x2, y2 = F2(x0, y0, r1, r2, eps)
            G = np.array([x2 - x0, y2 - y0])
            if np.linalg.norm(G) < tol:
                return x0, y0
            else:
                J = JacobianF2(x0, y0, r1, r2, eps) - np.eye(2)
                delta = np.linalg.solve(J, -G)
                x0 += delta[0]
                y0 += delta[1]
        elif p == 3:
            x2, y2 = F3(x0, y0, r1, r2, eps)
            G = np.array([x2 - x0, y2 - y0])
            if np.linalg.norm(G) < tol:
                return x0, y0
            else:
                J = JacobianF3(x0, y0, r1, r2, eps) - np.eye(2)
                delta = np.linalg.solve(J, -G)
                x0 += delta[0]
                y0 += delta[1]
        else:
            raise ValueError("please pick either 2- or 3-orbit")
    if np.linalg.norm(G) >= tol:
        print("no solution found within max tolerance")
        return None
    return x0, y0


def main():
    #sanity check, fixed points
    print(F(0, 0, 3.1, 3.4, 0.3))
    print(F2(0, 0, 3.1, 3.4, 0.3))
    print(F3(0, 0, 3.1, 300, 0.3))
if __name__ == "__main__":
    main()





























# """
# Exercise 2: Periodic orbits of period 2 and 3.

# Finds periodic points by solving G_p(x, y) = F^p(x, y) - (x, y) = 0
# using scipy.optimize.fsolve (Newton based).
# """

# import numpy as np
# from scipy.optimize import fsolve
# import matplotlib.pyplot as plt
# from coupled_map import coupled_map, compose

# # --- Parameters ---
# PARAM_SETS = [
#     (3.1, 3.4, 0.3),
#     (3.1, 3.55, 0.3),
#     (3.1, 3.8, 0.3),
# ]


# def G(xy, r1, r2, eps, period):
#     """Residual: F^p(x,y) - (x,y)."""
#     x, y = xy
#     xp, yp = compose(x, y, r1, r2, eps, period)
#     return [xp - x, yp - y]


# def find_periodic_points(r1, r2, eps, period, n_grid=20, tol=1e-10):
#     """Search for period-p points on a grid of initial guesses."""
#     solutions = []
#     for x0 in np.linspace(0.01, 0.99, n_grid):
#         for y0 in np.linspace(0.01, 0.99, n_grid):
#             sol, info, ier, msg = fsolve(
#                 G, [x0, y0], args=(r1, r2, eps, period), full_output=True
#             )
#             if ier == 1 and 0 < sol[0] < 1 and 0 < sol[1] < 1:
#                 # Check it's not a duplicate
#                 is_new = True
#                 for s in solutions:
#                     if np.linalg.norm(sol - s) < 1e-6:
#                         is_new = False
#                         break
#                 if is_new:
#                     solutions.append(sol)
#     return solutions


# def minimal_period(sol, r1, r2, eps, target_period):
#     """Check that the solution has exactly the target minimal period."""
#     x, y = sol
#     for p in range(1, target_period):
#         xp, yp = compose(x, y, r1, r2, eps, p)
#         if abs(xp - x) < 1e-8 and abs(yp - y) < 1e-8:
#             return p  # actual minimal period is smaller
#     return target_period


# def main():
#     for r1, r2, eps in PARAM_SETS:
#         print(f"\n{'='*60}")
#         print(f"Parameters: r1={r1}, r2={r2}, eps={eps}")
#         print(f"{'='*60}")

#         for period in [2, 3]:
#             sols = find_periodic_points(r1, r2, eps, period)
#             print(f"\n  Period-{period} candidates found: {len(sols)}")
#             for s in sols:
#                 mp = minimal_period(s, r1, r2, eps, period)
#                 tag = f"  (minimal period = {mp})" if mp < period else ""
#                 print(f"    ({s[0]:.10f}, {s[1]:.10f}){tag}")

#             # TODO: filter out solutions whose minimal period < target
#             # TODO: print the full orbit for each genuine periodic point


# if __name__ == "__main__":
#     main()
