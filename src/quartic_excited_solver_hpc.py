# Module: quartic_ground_solver_hpc
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np
from decimal import Decimal, getcontext
import math
import json
from src.quartic_excited_solver import quartic_excited_solver, save_results

def seq_quartic_excited_solver(c, E0s, truncations, precision, tol, max_iter=100, n_excited=5):
    """
    Function to solve the Riccati equation for the excited states of the quartic potential in sequence.
    
    Parameters:
    c (Decimal): The parameter of the transformation.
    E0s (list): The energy guesses for the first n excited states.
    truncations (int): The maximum truncation order for the expansion.
    precision (int): The precision for the high precision arithmetic.
    tol (Decimal): The tolerance for energy eigenvalue convergence.
    max_iter (int): The maximum number of iterations for the Newton method.
    n_excited (int): The number of excited states to solve.
    
    Returns:
    save data in json files.
    """
    getcontext().prec = precision
    
    c = Decimal(str(c))
    E0s = [Decimal(str(E0)) for E0 in E0s]
    tol = Decimal(str(tol))
    # update energy guess in each truncation order
    for truncation_order in range(10, truncations+1):
        Es, expansions = quartic_excited_solver(c, E0s, truncation_order, precision, tol, max_iter, n_excited)
        # save data
        # filename includes n_excited, c, truncation_order, precision, tol
        filename = f"results/quartic_excited_{n_excited}_c_{c}_trunc_{truncation_order}_prec_{precision}_tol_{tol}.json"
        save_results(filename, Es, expansions)
        # update energy guesses
        E0s = Es
    return None
