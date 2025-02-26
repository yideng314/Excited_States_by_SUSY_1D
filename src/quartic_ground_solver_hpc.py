# Module: quartic_ground_solver_hpc
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np
from decimal import Decimal, getcontext
import math
import json
from src.quartic_ground_solver import quartic_ground_solver

def seq_quartic_ground_solver(c, E0, truncations, precision, tol, max_iter=100):
    """
    Function to solve the Riccati equation for the ground state of the quartic potential in sequence.
    
    Parameters:
    c (Decimal): The parameter of the transformation.
    E0 (Decimal): The energy guess.
    truncations (int): The maximum truncation order for the Riccati equation.
    precision (int): The precision for the high precision arithmetic.
    tol (Decimal): The tolerance for energy eigenvalue convergence.
    max_iter (int): The maximum number of iterations for the Newton method.
    
    Returns:
    data saved in the json files by the quartic_ground_solver function.
    """
    getcontext().prec = precision
    
    c = Decimal(str(c))
    E0 = Decimal(str(E0))
    tol = Decimal(str(tol))
    # update energy guess in each truncation order
    for truncation_order in range(2, truncations+1):
        E0 = quartic_ground_solver(c, E0, truncation_order, precision, tol, max_iter)
    return None
