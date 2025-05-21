import numpy as np
import math
from decimal import Decimal, getcontext
import mpmath as mp
from functools import lru_cache
import json
import csv

mp.mp.dps = 500
getcontext().prec = 500
size = 262144*16

def load_excited_states_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    excited_states = []
    for item in data:
        energy = Decimal(item["energy"])
        state = np.array([Decimal(x) for x in item["state"]])
        excited_states.append((energy, state))
    
    return excited_states

def load(rank, n):
    file_name = f'excited_states_nmax_5_rank_{rank}_omega_8_tol_1E-400_prec_500.json'
    excited_states = load_excited_states_from_json(file_name)
    # 取出第n个state的能量和波函数
    energy, state = excited_states[n]
    return energy, state

def resi(rank, n, x):
    omega = Decimal('8')
    a = omega.sqrt()
    energy, V_decimal = load(rank, n)
    
    @lru_cache(maxsize=size)
    def H(n, x):
        if n == 0:
            return 1
        elif n == 1:
            return 2 * x
        else:
            return 2 * x * H(n - 1, x) - 2 * (n - 1) * H(n - 2, x)

    def exp(x):
        return Decimal(str(mp.exp(x)))

    pi = Decimal(str(mp.pi))

    @lru_cache(maxsize=size)
    def psi_1d(n, x):
        return exp(-a ** 2 * x ** 2 / 2) * H(n, a * x) * (a / (2 ** n * math.factorial(n) * pi.sqrt())).sqrt()

    def phi_x(x):
        value = Decimal(0)
        for i in range(len(V_decimal)):
            value += V_decimal[i] * psi_1d(i, x)
        return value

    def cal_resi(x):
        delta = Decimal('1e-50')
        laplacian = (phi_x(x + delta) - 2 * phi_x(x) + phi_x(x - delta)) / delta ** 2
        V = x**4 / 2
        out = -laplacian / 2 + V * phi_x(x) - energy * phi_x(x)
        return out

    return cal_resi(x)

if __name__ == "__main__":
    rank = 100
    
    file_name = f'basis_resi_rank_{rank}.csv'
    # Open a CSV file to write the header
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['x', 'n=0', 'n=1', 'n=2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=' ')
        writer.writeheader()
    
    x_values = [Decimal(i) for i in np.linspace(-5, 5, 100)]

    # Write the data row by row
    for x in x_values:
        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=' ')
            writer.writerow({
                'x': float(x),
                'n=0': float(resi(rank, 0, x)),
                'n=1': float(resi(rank, 1, x)),
                'n=2': float(resi(rank, 2, x))
            })
