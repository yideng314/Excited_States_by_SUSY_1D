# Module: riccati_equation_solver
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np
import math
from decimal import Decimal, getcontext
import json

def recurrence_with_derivative_solver(f1, f2, f3, f4, f5, expansion, d_expansion, l, E, precision=50):
    """
    Function to solve the recurrence relation of the l-th taylor coefficient of the solution and its derivative in the Riccati equation.
    
    Parameters:
    f1, f2, f3, f4, f5 (np.array): The taylor coefficients array of the Riccati equation.
    expansion (np.array): The taylor coefficients list of the solution.
    d_expansion (np.array): The taylor coefficients list of the derivative of the solution.
    l (int): The index of the taylor coefficient to be determined.
    E (Decimal): The energy guess to define the Riccati equation.
    truncation_order (int): The truncation order of the solution, which is the length of the taylor coefficients list.
    precision (int): The precision for the high precision arithmetic.
    
    Returns:
    expansion[l] (Decimal): The l-th taylor coefficient of the solution.
    d_expansion[l] (Decimal): The l-th taylor coefficient of the derivative of the solution.
    """
    getcontext().prec = precision
    
    # get the length of the taylor coefficients list
    d1, d2, d3, d4, d5 = len(f1), len(f2), len(f3), len(f4), len(f5)
    # initialize the recurrence and its derivative
    recurrence = Decimal(0)
    d_recurrence = Decimal(0)
    # f1-related term
    recurrence += sum((k+1)*f1[l-k]*expansion[k+1] for k in range(max(0,l-d1+1),l)) # from max(0,l-d1) to l-1
    d_recurrence += sum((k+1)*f1[l-k]*d_expansion[k+1] for k in range(max(0,l-d1+1),l)) # from max(0,l-d1) to l-1
    # f2-related term
    recurrence += sum(f2[l-k]*sum(expansion[p]*expansion[k-p] for p in range(0,k+1)) for k in range(max(0,l-d2+1),l)) # f2 from max(0,l-d2) to l-1, expansion from 0 to k
    d_recurrence += 2*sum(f2[l-k]*sum(d_expansion[p]*expansion[k-p] for p in range(0,k+1)) for k in range(max(0,l-d2+1),l)) # f2 from max(0,l-d2) to l-1, expansion from 0 to k
    # f3-related term
    recurrence += sum(f3[l-k]*expansion[k] for k in range(max(0,l-d3+1),l+1)) # from max(0,l-d3) to l
    d_recurrence += sum(f3[l-k]*d_expansion[k] for k in range(max(0,l-d3+1),l+1)) # from max(0,l-d3) to l
    # f4-related term
    if l < d4:
        recurrence += f4[l]
    # f5-related term
    if l < d5:
        recurrence += -f5[l]*E
        d_recurrence += -f5[l]
    # divide by -(l*f1[1]+f3[0])
    recurrence /= -(l*f1[1]+f3[0])
    d_recurrence /= -(l*f1[1]+f3[0])
    return recurrence, d_recurrence

def expansion_with_derivative_solver(f1, f2, f3, f4, f5, E, truncation_order, precision=50):
    """
    Function to generate the taylor coefficients list of the solution and its derivative in the Riccati equation.
    
    Parameters:
    f1, f2, f3, f4, f5 (np.array): The taylor coefficients list of the Riccati equation.
    E (Decimal): The energy guess to define the Riccati equation.
    truncation_order (int): The truncation order of the solution, which is the length of the taylor coefficients list.
    precision (int): The precision for the high precision arithmetic.
    
    Returns:
    expansion (np.array): The taylor coefficients list of the solution.
    d_expansion (np.array): The taylor coefficients list of the derivative of the solution.
    """
    getcontext().prec = precision
    
    # initialize the expansion and its derivative
    expansion = np.zeros(truncation_order, dtype=Decimal)
    d_expansion = np.zeros(truncation_order, dtype=Decimal)
    # determine the expansion and its derivative coefficients.
    for i in range(truncation_order):
        expansion[i], d_expansion[i] = recurrence_with_derivative_solver(f1, f2, f3, f4, f5, expansion, d_expansion, i, E, precision)
    return expansion, d_expansion

def newton_method(f1, f2, f3, f4, f5, E0, truncation_order, precision=50, max_iter=100, tol=1e-10):
    """
    Function to solve the Riccati equation using Newton method.
    
    Parameters:
    f1, f2, f3, f4, f5 (np.array): The taylor coefficients list of the Riccati equation.
    E0 (Decimal): The initial energy guess to define the Riccati equation.
    truncation_order (int): The truncation order of the solution, which is the length of the taylor coefficients list.
    precision (int): The precision for the high precision arithmetic.
    max_iter (int): The maximum number of iterations for the Newton method.
    tol (float): The tolerance for energy eigenvalue convergence.
    
    Returns:
    E (Decimal): The energy eigenvalue.
    expansion (np.array): The taylor coefficients list of the solution.
    """
    getcontext().prec = precision
    # convert the taylor coefficients array to Decimal type
    f1 = np.array([Decimal(str(f1[j])) for j in range(len(f1))])
    f2 = np.array([Decimal(str(f2[j])) for j in range(len(f2))])
    f3 = np.array([Decimal(str(f3[j])) for j in range(len(f3))])
    f4 = np.array([Decimal(str(f4[j])) for j in range(len(f4))])
    f5 = np.array([Decimal(str(f5[j])) for j in range(len(f5))])
    # convert the energy guess to Decimal type
    E0 = Decimal(str(E0))
    # initialize the energy eigenvalue
    E = E0
    # tol converted to Decimal type
    tol = Decimal(str(tol))
    # iteration and counter
    for i in range(max_iter):
        # generate the expansion and its derivative
        expansion, d_expansion = expansion_with_derivative_solver(f1, f2, f3, f4, f5, E, truncation_order, precision)
        # calculate the residue: sum(expansion)+sum(f4).sqrt()==0
        residue = sum(expansion) + sum(f4).sqrt()
        # calculate the derivative of the residue
        d_residue = sum(d_expansion)
        # convergence check
        diff = -residue/d_residue
        print(f"Newton method iteration {i+1}, diff = {diff:.2e}.")
        if abs(diff) < tol: # print diff with scientific notation
            print(f"Newton method converges at iteration {i+1}, with energy convergence {diff:.2e}.")
            break
        # update the energy eigenvalue
        E += diff
    return E, expansion

def save_energy_and_expansion_to_json(filename, E, expansion):
    """
    Function to save the energy eigenvalue and the taylor coefficients list of the solution to a JSON file.
    
    Parameters:
    filename (str): The name of the JSON file.
    E (Decimal): The energy eigenvalue.
    expansion (np.array): The taylor coefficients list of the solution.
    
    Returns:
    None
    """
    data = {
        "energy": str(E),
        "expansion": [str(x) for x in expansion]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
def load_energy_and_expansion_from_json(filename):
    """
    Function to load the energy eigenvalue and the taylor coefficients list of the solution from a JSON file.
    
    Parameters:
    filename (str): The name of the JSON file.
    
    Returns:
    E (Decimal): The energy eigenvalue.
    expansion (np.array): The taylor coefficients list of the solution.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    E = Decimal(data["energy"])
    expansion = np.array([Decimal(x) for x in data["expansion"]])
    return E, expansion
