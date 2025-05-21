# Module: wave_function
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np
import math
from decimal import Decimal, getcontext
import json
import mpmath
import csv

# set the precision
getcontext().prec = 400
mpmath.mp.dps = 400

# load the results from JSON file
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

# quartic_excited_5_c_1_trunc_10_prec_300_tol_1E-200.json
# quartic_excited_5_c_5_trunc_270_prec_400_tol_1E-300.json
def coefficients(c, truncation_order, n_excited, precision=400, tol='1E-300'):
    """
    Function to load the energy eigenvalues and expansions from a JSON file.
    
    Parameters:
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    n_excited (int): The value of the excited state.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    E (Decimal): The energy eigenvalue.
    expansion (list): The expansion.
    """
    filename = f"quartic_excited_5_c_{c}_trunc_{truncation_order}_prec_{precision}_tol_{tol}.json"
    Es, expansions = load_results(filename)
    expansion = expansions[n_excited]
    return Es, expansion

def exp(x):
    return Decimal(str(mpmath.exp(x)))

def psi_ground(x, c, truncation_order, n_excited, precision=400, tol='1E-300'):
    """
    Function to calculate the wave function of the ground state.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    n_excited (int): The value of the excited state.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    _, expansion = coefficients(c, truncation_order, n_excited, precision, tol)
    
    # F_z
    def F_z(z):
        out1 = -sum(expansion[l]*sum(int(math.comb(l,i))*(-1)**i*(1-z)**(i-3)/(i-3) for i in range(4,l+1)) for l in range(4,len(expansion)))
        out2 = -sum(expansion[l]*(Decimal(1)/3-Decimal(l)/2+Decimal(l*(l-1))/2-sum(Decimal(math.comb(l,i))*(-1)**i/(i-3) for i in range(4,l+1))) for l in range(len(expansion)))
        return out1+out2
    # psi
    z = 1-(c/(c+x**2)).sqrt()
    F = F_z(z)
    # psi = psi_z(z)
    return F

def psi_1_0(x, c, truncation_order, precision=300, tol='1E-200'):
    """
    Function to calculate the (10) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    return psi_ground(x, c, truncation_order, 0, precision, tol)

def psi_2_0(x, c, truncation_order, precision=300, tol='1E-200'):
    """
    Function to calculate the (20) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    return psi_ground(x, c, truncation_order, 1, precision, tol)

def psi_3_0(x, c, truncation_order, precision=300, tol='1E-200'):
    """
    Function to calculate the (30) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    return psi_ground(x, c, truncation_order, 2, precision, tol)

if __name__ == "__main__":    
    # x values from -5 to 5, decimal type
    x_values = [Decimal(i) for i in np.linspace(-5, 5, 100)]
    c = 5
    # write the truncation order
    truncation_order = 800

    file_name = f'F_{c}_trunc_{truncation_order}.csv'
    # Open a CSV file to write the header
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['x', 'F_1_0', 'F_2_0', 'F_3_0']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=' ')
        writer.writeheader()

    # Write the data row by row
    for x in x_values:
        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=' ')
            writer.writerow({
                'x': x,
                'F_1_0': float(psi_1_0(x, c, truncation_order)),
                'F_2_0': float(psi_2_0(x, c, truncation_order)),
                'F_3_0': float(psi_3_0(x, c, truncation_order))
            })
