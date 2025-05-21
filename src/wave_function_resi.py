# Module: wave_function_resi
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
    
    # alpha, gamma, zeta, eta
    alpha = -sum(expansion[k] for k in range(len(expansion)))/3
    gamma = -sum(expansion[k]*k for k in range(1,len(expansion)))/2
    zeta = sum(expansion[k]*k*(k-1) for k in range(2,len(expansion)))/2
    eta = sum(expansion[k]*k*(k-1)*(k-2) for k in range(3,len(expansion)))/6
    # F_z
    def F_z(z):
        out1 = -sum(expansion[l]*sum(int(math.comb(l,i))*(-1)**i*(1-z)**(i-3)/(i-3) for i in range(4,l+1)) for l in range(4,len(expansion)))
        out2 = -sum(expansion[l]*(Decimal(1)/3-Decimal(l)/2+Decimal(l*(l-1))/2-sum(Decimal(math.comb(l,i))*(-1)**i/(i-3) for i in range(4,l+1))) for l in range(len(expansion)))
        return out1+out2
    # psi_z
    def psi_z(z):
        part1 = (1-z)**eta 
        part2 = exp(-alpha/(1-z)**3+gamma/(1-z)**2+zeta/(1-z))
        part3 = exp(F_z(z))
        return part1*part2*part3
    # psi
    z = 1-(c/(c+x**2)).sqrt()
    psi = psi_z(z)
    return psi

def psi_1_0(x, c, truncation_order, precision=400, tol='1E-300'):
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

def psi_2_0(x, c, truncation_order, precision=400, tol='1E-300'):
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

def psi_3_0(x, c, truncation_order, precision=400, tol='1E-300'):
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

def psi_4_0(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (40) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    return psi_ground(x, c, truncation_order, 3, precision, tol)

def psi_5_0(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (50) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    return psi_ground(x, c, truncation_order, 4, precision, tol)

def psi_1_1(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (11) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    Es, expansion = coefficients(c, truncation_order, 1, precision, tol)
    N = 1/(2*(Es[1]-Es[0])).sqrt()
    h = Decimal('1E-100')
    d_psi_2_0 = (psi_2_0(x+h, c, truncation_order)-psi_2_0(x-h, c, truncation_order))/(2*h)
    d_psi_1_0 = (psi_1_0(x+h, c, truncation_order)-psi_1_0(x-h, c, truncation_order))/(2*h)
    psi = (d_psi_2_0 + d_psi_1_0*psi_2_0(x, c, truncation_order)/psi_1_0(x, c, truncation_order))
    return N*psi

def psi_2_1(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (21) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    Es, expansion = coefficients(c, truncation_order, 2, precision, tol)
    N = 1/(2*(Es[2]-Es[1])).sqrt()
    h = Decimal('1E-100')
    d_psi_3_0 = (psi_3_0(x+h, c, truncation_order)-psi_3_0(x-h, c, truncation_order))/(2*h)
    d_psi_2_0 = (psi_2_0(x+h, c, truncation_order)-psi_2_0(x-h, c, truncation_order))/(2*h)
    psi = (d_psi_3_0 + d_psi_2_0*psi_3_0(x, c, truncation_order)/psi_2_0(x, c, truncation_order))
    return N*psi

def psi_1_2(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (12) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    Es, expansion = coefficients(c, truncation_order, 2, precision, tol)
    N = 1/(2*(Es[2]-Es[0])).sqrt()
    h = Decimal('1E-100')
    d_psi_2_1 = (psi_2_1(x+h, c, truncation_order)-psi_2_1(x-h, c, truncation_order))/(2*h)
    d_psi_1_0 = (psi_1_0(x+h, c, truncation_order)-psi_1_0(x-h, c, truncation_order))/(2*h)
    psi = (d_psi_2_1 + d_psi_1_0*psi_2_1(x, c, truncation_order)/psi_1_0(x, c, truncation_order))
    return N*psi

def psi_3_1(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (31) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    Es, expansion = coefficients(c, truncation_order, 3, precision, tol)
    N = 1/(2*(Es[3]-Es[2])).sqrt()
    h = Decimal('1E-100')
    d_psi_4_0 = (psi_4_0(x+h, c, truncation_order)-psi_4_0(x-h, c, truncation_order))/(2*h)
    d_psi_3_0 = (psi_3_0(x+h, c, truncation_order)-psi_3_0(x-h, c, truncation_order))/(2*h)
    psi = (d_psi_4_0 + d_psi_3_0*psi_4_0(x, c, truncation_order)/psi_3_0(x, c, truncation_order))
    return N*psi

def psi_2_2(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (22) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    Es, expansion = coefficients(c, truncation_order, 3, precision, tol)
    N = 1/(2*(Es[3]-Es[2])).sqrt()
    h = Decimal('1E-100')
    d_psi_3_1 = (psi_3_1(x+h, c, truncation_order)-psi_3_1(x-h, c, truncation_order))/(2*h)
    d_psi_2_0 = (psi_2_0(x+h, c, truncation_order)-psi_2_0(x-h, c, truncation_order))/(2*h)
    psi = (d_psi_3_1 + d_psi_2_0*psi_3_1(x, c, truncation_order)/psi_2_0(x, c, truncation_order))
    return N*psi

def psi_1_3(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (13) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    Es, expansion = coefficients(c, truncation_order, 3, precision, tol)
    N = 1/(2*(Es[3]-Es[0])).sqrt()
    h = Decimal('1E-100')
    d_psi_2_2 = (psi_2_2(x+h, c, truncation_order)-psi_2_2(x-h, c, truncation_order))/(2*h)
    d_psi_1_0 = (psi_1_0(x+h, c, truncation_order)-psi_1_0(x-h, c, truncation_order))/(2*h)
    psi = (d_psi_2_2 + d_psi_1_0*psi_2_2(x, c, truncation_order)/psi_1_0(x, c, truncation_order))
    return N*psi

def psi_4_1(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (41) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    Es, expansion = coefficients(c, truncation_order, 4, precision, tol)
    N = 1/(2*(Es[4]-Es[3])).sqrt()
    h = Decimal('1E-100')
    d_psi_5_0 = (psi_5_0(x+h, c, truncation_order)-psi_5_0(x-h, c, truncation_order))/(2*h)
    d_psi_4_0 = (psi_4_0(x+h, c, truncation_order)-psi_4_0(x-h, c, truncation_order))/(2*h)
    psi = (d_psi_5_0 + d_psi_4_0*psi_5_0(x, c, truncation_order)/psi_4_0(x, c, truncation_order))
    return N*psi

def psi_3_2(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (32) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    Es, expansion = coefficients(c, truncation_order, 4, precision, tol)
    N = 1/(2*(Es[4]-Es[2])).sqrt()
    h = Decimal('1E-100')
    d_psi_4_1 = (psi_4_1(x+h, c, truncation_order)-psi_4_1(x-h, c, truncation_order))/(2*h)
    d_psi_3_0 = (psi_3_0(x+h, c, truncation_order)-psi_3_0(x-h, c, truncation_order))/(2*h)
    psi = (d_psi_4_1 + d_psi_3_0*psi_4_1(x, c, truncation_order)/psi_3_0(x, c, truncation_order))
    return N*psi

def psi_2_3(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (23) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    Es, expansion = coefficients(c, truncation_order, 4, precision, tol)
    N = 1/(2*(Es[4]-Es[1])).sqrt()
    h = Decimal('1E-100')
    d_psi_3_2 = (psi_3_2(x+h, c, truncation_order)-psi_3_2(x-h, c, truncation_order))/(2*h)
    d_psi_2_0 = (psi_2_0(x+h, c, truncation_order)-psi_2_0(x-h, c, truncation_order))/(2*h)
    psi = (d_psi_3_2 + d_psi_2_0*psi_3_2(x, c, truncation_order)/psi_2_0(x, c, truncation_order))
    return N*psi

def psi_1_4(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the (14) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The value of the truncation order.
    precision (int): The value of the precision.
    tol (str): The value of the tolerance.
    
    Returns:
    psi (Decimal): The value of the wave function.
    """
    Es, expansion = coefficients(c, truncation_order, 4, precision, tol)
    N = 1/(2*(Es[4]-Es[0])).sqrt()
    h = Decimal('1E-100')
    d_psi_2_3 = (psi_2_3(x+h, c, truncation_order)-psi_2_3(x-h, c, truncation_order))/(2*h)
    d_psi_1_0 = (psi_1_0(x+h, c, truncation_order)-psi_1_0(x-h, c, truncation_order))/(2*h)
    psi = (d_psi_2_3 + d_psi_1_0*psi_2_3(x, c, truncation_order)/psi_1_0(x, c, truncation_order))
    return N*psi

def resi_1_0(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the residual of the (10) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The truncation order of the wave function.
    precision (int): The precision of the calculation.
    tol (str): The tolerance of the calculation.
    
    Returns:
    resi (Decimal): The residual of the wave function.
    """
    # energy
    Es, expansion = coefficients(c, truncation_order, 0, precision, tol)
    gs_energy = Es[0]
    # wave function
    gs_wave = psi_1_0(x, c, truncation_order, precision, tol)
    # second order derivative of the wave function, using numerical differentiation
    h = Decimal('1E-80')
    psi_1 = psi_1_0(x-h, c, truncation_order, precision, tol)
    psi_2 = psi_1_0(x+h, c, truncation_order, precision, tol)
    psi_2nd = (psi_2 - 2*gs_wave + psi_1) / h**2
    # residual = -1/2 * psi'' + 1/2 * x^4 * psi - E * psi
    resi = -psi_2nd/2 + x**4/2 * gs_wave - gs_energy * gs_wave
    return resi

def resi_1_1(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the residual of the (11) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The truncation order of the wave function.
    precision (int): The precision of the calculation.
    tol (str): The tolerance of the calculation.
    
    Returns:
    resi (Decimal): The residual of the wave function.
    """
    # energy
    Es, expansion = coefficients(c, truncation_order, 0, precision, tol)
    first_energy = Es[1]
    # wave function
    first_wave = psi_1_1(x, c, truncation_order, precision, tol)
    # second order derivative of the wave function, using numerical differentiation
    h = Decimal('1E-80')
    psi_1 = psi_1_1(x-h, c, truncation_order, precision, tol)
    psi_2 = psi_1_1(x+h, c, truncation_order, precision, tol)
    psi_2nd = (psi_2 - 2*first_wave + psi_1) / h**2
    # residual = -1/2 * psi'' + 1/2 * x^4 * psi - E * psi
    resi = -psi_2nd/2 + x**4/2 * first_wave - first_energy * first_wave
    return resi

def resi_1_2(x, c, truncation_order, precision=400, tol='1E-300'):
    """
    Function to calculate the residual of the (12) wave function.
    
    Parameters:
    x (Decimal): The value of the position.
    c (int): The value of the parameter c.
    truncation_order (int): The truncation order of the wave function.
    precision (int): The precision of the calculation.
    tol (str): The tolerance of the calculation.
    
    Returns:
    resi (Decimal): The residual of the wave function.
    """
    # energy
    Es, expansion = coefficients(c, truncation_order, 0, precision, tol)
    second_energy = Es[2]
    # wave function
    second_wave = psi_1_2(x, c, truncation_order, precision, tol)
    # second order derivative of the wave function, using numerical differentiation
    h = Decimal('1E-80')
    psi_1 = psi_1_2(x-h, c, truncation_order, precision, tol)
    psi_2 = psi_1_2(x+h, c, truncation_order, precision, tol)
    psi_2nd = (psi_2 - 2*second_wave + psi_1) / h**2
    # residual = -1/2 * psi'' + 1/2 * x^4 * psi - E * psi
    resi = -psi_2nd/2 + x**4/2 * second_wave - second_energy * second_wave
    return resi

if __name__ == '__main__':
    # x values from -5 to 5, decimal type
    x_values = [Decimal(str(i)) for i in np.linspace(-8, 8, 100)]
    c = 5
    
    truncation_order = 100
    
    file_name = f'wf_resi_{c}_trunc_{truncation_order}.csv'
    # Open a CSV file to write the header
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['x', 'resi_1_0', 'resi_1_1', 'resi_1_2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=' ')
        writer.writeheader()

    # Write the data row by row
    for x in x_values:
        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=' ')
            writer.writerow({
                'x': x,
                'resi_1_0': float(resi_1_0(x, c, truncation_order)),
                'resi_1_1': float(resi_1_1(x, c, truncation_order)),
                'resi_1_2': float(resi_1_2(x, c, truncation_order))
            })
