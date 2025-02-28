import numpy as np
from decimal import Decimal, getcontext
import math
import json
from src.quartic_excited_solver import quartic_excited_solver, save_results
from src.quartic_excited_solver_hpc import *

if __name__ == "__main__":
    c = Decimal('2')
    E0s = [Decimal('0.53'), Decimal('1.90'), Decimal('3.72'), Decimal('5.82'), Decimal('8.13')]
    truncations = 100
    precision = 200
    tol = Decimal('1e-100')
    seq_quartic_excited_solver(c, E0s, truncations, precision, tol, max_iter=100, n_excited=5)
    print("Done!")

# Output:
# 
# Newton method iteration 1, diff = 1.81e-4.
# Newton method iteration 2, diff = -3.54e-7.
# Newton method iteration 3, diff = -1.36e-12.
# Newton method iteration 4, diff = -1.99e-23.
# Newton method iteration 5, diff = -4.27e-45.
# Newton method iteration 6, diff = -1.97e-88.
# Newton method iteration 7, diff = -4.19e-175.
# Newton method converges at iteration 7, with energy convergence -4.19e-175.
# Newton method iteration 1, diff = -1.64e-4.
# Newton method iteration 2, diff = -3.66e-7.
# Newton method iteration 3, diff = -1.81e-12.
# Newton method iteration 4, diff = -4.42e-23.
# Newton method iteration 5, diff = -2.64e-44.
# Newton method iteration 6, diff = -9.44e-87.
# Newton method iteration 7, diff = -1.20e-171.
# Newton method converges at iteration 7, with energy convergence -1.20e-171.
# ...
