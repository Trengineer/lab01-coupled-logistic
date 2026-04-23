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

def newton2D(x0, y0, p, r1, r2, eps, tol=1e-10, max_iter=50):
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
                if not (0 <= x0 <= 1 and 0 <= y0 <= 1):
                    return None
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
                if not (0 <= x0 <= 1 and 0 <= y0 <= 1):
                    return None
        else:
            raise ValueError("please pick either 2- or 3-orbit")
    if np.linalg.norm(G) >= tol:
        return None
    return x0, y0

def orbit_is_physical(x, y, p, r1, r2, eps):
    for _ in range(p):
        x, y = F(x, y, r1, r2, eps)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return False
    return True

def is_minimal_period(x, y, p, r1, r2, eps, tol=1e-6):
    # check its not a fixed point
    x1, y1 = F(x, y, r1, r2, eps)
    if np.sqrt((x1-x)**2 + (y1-y)**2) < tol:
        return False
    # for period 3, also check its not period-2
    if p == 3:
        x2, y2 = F2(x, y, r1, r2, eps)
        if np.sqrt((x2-x)**2 + (y2-y)**2) < tol:
            return False
    return True

def find_periodic_orbit(p, r1, r2, eps, n_guesses=50):
    results = []
    values = np.linspace(0, 1, n_guesses)
    for x0 in values:
        for y0 in values:
            solution = newton2D(x0, y0, p, r1, r2, eps, tol=1e-8, max_iter=100)
            if solution is not None:
                if 0 <= solution[0] <= 1 and 0 <= solution[1] <= 1:
                    if is_minimal_period(solution[0], solution[1], p, r1, r2, eps, tol=1e-6): #look if the period isnt another period
                        if orbit_is_physical(solution[0], solution[1], p, r1, r2, eps): #check that intermediate coordinates/populations is inside [0,1]
                            results.append((x0, y0, solution))

    unique = []
    for sol in results:
        is_duplicate = False
        for existing in unique:
            if np.sqrt((sol[2][0] - existing[2][0])**2 + (sol[2][1] - existing[2][1])**2) < 1e-6:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(sol)

    return unique

def main():
    param_sets = [
        (3.1, 3.4, 0.3),
        (3.1, 3.55, 0.3),
        (3.1, 3.8, 0.3),
        (3.1, 5, 0.3),
    ]

    for p in [2, 3]:
        for r1, r2, eps in param_sets:
            print(f"======== Period-{p} orbits for (r1, r2, ε) = ({r1}, {r2}, {eps}) =========")
            orbits = find_periodic_orbit(p=p, r1=r1, r2=r2, eps=eps)
            for x0_guess, y0_guess, sol in orbits:
                print(f"  solution {sol} found from initial guess ({x0_guess:.4f}, {y0_guess:.4f})")

    print("\n------Verification of period-2 orbits-------")
    for r1, r2, eps in param_sets:
        print(f"  (r1, r2, ε) = ({r1}, {r2}, {eps})")
        orbits = find_periodic_orbit(p=2, r1=r1, r2=r2, eps=eps, n_guesses=50)
        for i, (x0_guess, y0_guess, sol) in enumerate(orbits):
            x, y = sol
            x1, y1 = F(x, y, r1, r2, eps)
            x2, y2 = F2(x, y, r1, r2, eps)
            print(f"    Point {i}: ({x:.6f}, {y:.6f})")
            print(f"      F  -> ({x1:.6f}, {y1:.6f})")
            print(f"      F2 -> ({x2:.6f}, {y2:.6f})  (should match point {i})")
        print()

    print("------Verification of period-3 orbits-------")
    for r1, r2, eps in param_sets:
        print(f"  (r1, r2, ε) = ({r1}, {r2}, {eps})")
        orbits = find_periodic_orbit(p=3, r1=r1, r2=r2, eps=eps, n_guesses=50)
        for i, (x0_guess, y0_guess, sol) in enumerate(orbits):
            x, y = sol
            x1, y1 = F(x, y, r1, r2, eps)
            x2, y2 = F2(x, y, r1, r2, eps)
            x3, y3 = F3(x, y, r1, r2, eps)
            print(f"    Point {i}: ({x:.6f}, {y:.6f})")
            print(f"      F  -> ({x1:.6f}, {y1:.6f})")
            print(f"      F2 -> ({x2:.6f}, {y2:.6f})")
            print(f"      F3 -> ({x3:.6f}, {y3:.6f})  (should match point {i})")
        print()
if __name__ == "__main__":
    main()