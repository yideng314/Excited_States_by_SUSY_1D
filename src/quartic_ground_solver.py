# Module: quartic_ground_solver
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np
from decimal import Decimal, getcontext
import json
from src.riccati_equation_solver import newton_method

def quartic_ground_solver(c, E0, truncation_order, precision, tol=1e-10, max_iter=100):
    """
    Function to solve the Riccati equation for the ground state of the quartic potential.
    
    Parameters:
    c (Decimal): The parameter of the transformation.
    E0 (Decimal): The energy guess.
    truncation_order (int): The truncation order.
    precision (int): The precision for the high precision arithmetic.
    max_iter (int): The maximum number of iterations for the Newton method.
    tol (Decimal): The tolerance for energy eigenvalue convergence.
    
    Returns:
    E (Decimal): The energy eigenvalue in this truncation order.
    """
    getcontext().prec = precision
    
    c = Decimal(str(c))
    # define f1, f2, f3, f4, f5 based on the parameter c
    # the list will convert to Decimal type in the newton_method function.
    f1 = np.array([0, -2, 9, -16, 14, -6, 1])
    f2 = np.array([0, -2, 1])
    f3 = np.array([-1, 1, 4, -8, 5, -1])
    f4 = np.array([0, 0, 4, -4, 1])*c**3
    f5 = np.array([2, -8, 12, -8, 2])*c
    # set the initial energy guess
    E0 = Decimal(str(E0))
    tol = Decimal(str(tol))
    E, expansion = newton_method(f1, f2, f3, f4, f5, E0, truncation_order, precision, max_iter, tol)
    # save energy eigenvalue and the taylor coefficients list of the solution
    data = {
        "energy": str(E),
        "expansion": [str(x) for x in expansion]
    }
    filename = f"results/quartic_ground_state_c_{c}_truncation_order_{truncation_order}_precision_{precision}_tol_{tol}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    return E
    