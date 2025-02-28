# Module: quartic_excited_solver
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np
import math
from decimal import Decimal, getcontext
from src.riccati_equation_solver import newton_method
import json

# add arrays element-wise
def add_arrays(arr1, arr2):
    """
    Function to add two numpy arrays element-wise with different lengths.
    
    Parameters:
    arr1 (numpy.ndarray): The first array.
    arr2 (numpy.ndarray): The second array.
    
    Returns:
    numpy.ndarray: The element-wise sum of the two arrays.
    """
    # determine the length of the result
    result_len = max(len(arr1), len(arr2))
    # create the result array
    result = np.zeros(result_len, dtype=object)
    # add the two arrays element-wise
    result[:len(arr1)] += arr1
    result[:len(arr2)] += arr2
    return result

# convolve arrays
def convolve_arrays(arr1, arr2):
    """
    Function to perform convolution of two arrays.
    
    Parameters:
    arr1 (list of Decimal): The first array.
    arr2 (list of Decimal): The second array.
    
    Returns:
    list of Decimal: The convolution of the two arrays.
    """
    len1 = len(arr1)
    len2 = len(arr2)
    result_length = len1 + len2 - 1
    result = np.zeros(result_length, dtype=object)
    
    for i in range(len1):
        result[i:i+len2] += arr1[i] * arr2
    return result

# excited once solver
# energy guess for first five states of the quartic potential
# [0.53, 1.90, 3.72, 5.82, 8.13]

def excited_once_solver(f1, f2, f3, f4, f5, E0, c, truncation_order, precision, max_iter, tol):
    """
    Function to solve the current Riccati equation and get the new f4 for the next excited state.
    
    Parameters:
    f1, f2, f3, f4, f5 (np.array): The taylor coefficients list of the Riccati equation.
    E0 (Decimal): The initial energy guess to define the Riccati equation.
    c (Decimal): The constant c for transformation.
    truncation_order (int): The truncation order of the solution, which is the length of the taylor coefficients list.
    precision (int): The precision for the high precision arithmetic.
    max_iter (int): The maximum number of iterations for the Newton method.
    tol (Decimal): The tolerance for energy eigenvalue convergence.
    
    Returns:
    E (Decimal): The energy eigenvalue for the present state.
    expansion (np.array): The taylor coefficients list of the present state.
    f4 (np.array): The taylor coefficients list of the Riccati equation for the excited state.
    """
    # get energy and expansion for this state
    E, expansion = newton_method(f1, f2, f3, f4, f5, E0, truncation_order, precision, max_iter, tol)
    # get the new f4
    array1 = np.array([1,-4,6,-4,1])*4*c*E
    array2 = np.array([0,-2,1])*(-2)
    # update f4
    f4 = add_arrays(-f4, array1)
    f4 = add_arrays(f4, convolve_arrays(array2, convolve_arrays(expansion, expansion)))
    return E, expansion, f4

def quartic_excited_solver(c, E0s, truncation_order, precision, tol, max_iter=100, n_excited=5):
    """
    Function to solve the first n excited states of the quartic potential.
    
    Parameters:
    c (Decimal): The parameter of the transformation.
    E0s (list): The energy guesses for the first n excited states.
    truncation_order (int): The truncation order.
    precision (int): The precision for the high precision arithmetic.
    tol (str): The tolerance for energy eigenvalue convergence, which is a string to avoid the loss of precision.
    max_iter (int): The maximum number of iterations for the Newton method.
    n_excited (int): The number of excited states to solve.
    
    Returns:
    Es (list): The energy eigenvalues of the first n excited states.
    expansions (list): The taylor coefficients list of the solutions.
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
    # set the initial energy guesses
    E0s = [Decimal(str(E0)) for E0 in E0s]
    tol = Decimal(str(tol))
    Es = []
    expansions = []
    f4_list = []
    for i in range(n_excited):
        # update f4 to get the new excited state solution
        f4_list.append(f4)
        E, expansion, f4 = excited_once_solver(f1, f2, f3, f4, f5, E0s[i], c, truncation_order, precision, max_iter, tol)
        Es.append(E)
        expansions.append(expansion)
    # save f4 list to a JSON file
    filename = f"results/f4_c_{c}_trunc_{truncation_order}_prec_{precision}_tol_{tol}.json"
    save_f4(filename, f4_list)
    return Es, expansions

# save and load results
def save_results(filename, Es, expansions):
    """
    Function to save the energy eigenvalues and expansions to a JSON file.
    
    Parameters:
    filename (str): The name of the file to save the results.
    Es (list): The list of energy eigenvalues.
    expansions (list): The list of expansions.
    
    Returns:
    None
    """
    data = {
        "energies": [str(E) for E in Es],
        "expansions": [[str(x) for x in expansion] for expansion in expansions]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# save f4 in f4_list (length=n_excited) to a JSON file
def save_f4(filename, f4_list):
    """
    Function to save the f4 lists to a JSON file.
    
    Parameters:
    filename (str): The name of the file to save the f4 lists.
    f4_list (list): The list of f4 arrays.
    
    Returns:
    None
    """
    data = {
        "f4": [[str(x) for x in f4] for f4 in f4_list]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# load f4 from a JSON file
def load_f4(filename):
    """
    Function to load the f4 lists from a JSON file.
    
    Parameters:
    filename (str): The name of the file to load the f4 lists.
    
    Returns:
    f4_list (list): The list of f4 arrays.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    f4_list = [[Decimal(x) for x in f4] for f4 in data["f4"]]
    return f4_list

def load_results(filename):
    """
    Function to load the energy eigenvalues and expansions from a JSON file.
    
    Parameters:
    filename (str): The name of the file to load the results.
    
    Returns:
    Es (list): The list of energy eigenvalues.
    expansions (list): The list of expansions.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    Es = [Decimal(E) for E in data["energies"]]
    expansions = [[Decimal(x) for x in expansion] for expansion in data["expansions"]]
    return Es, expansions
