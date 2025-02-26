import numpy as np
from decimal import Decimal, getcontext
import math
import json
from src.quartic_excited_solver import quartic_excited_solver, save_results
from src.quartic_excited_solver_hpc import *

if __name__ == "__main__":
    c = Decimal('2')
    E0s = [Decimal('0.53'), Decimal('1.90'), Decimal('3.72'), Decimal('5.82'), Decimal('8.13')]
    truncations = 20
    precision = 200
    tol = Decimal('1e-100')
    seq_quartic_excited_solver(c, E0s, truncations, precision, tol, max_iter=100, n_excited=5)
    print("Done!")

# Output:
# 
# Newton method iteration 1, diff = -1.20e-1.
# Newton method iteration 2, diff = -3.23e-2.
# Newton method iteration 3, diff = -2.74e-3.
# Newton method iteration 4, diff = -2.01e-5.
# Newton method iteration 5, diff = -1.07e-9.
# Newton method iteration 6, diff = -3.07e-18.
# Newton method iteration 7, diff = -2.52e-35.
# Newton method iteration 8, diff = -1.70e-69.
# Newton method iteration 9, diff = -7.66e-138.
# Newton method converges at iteration 9, with energy convergence -7.66e-138.
# Newton method iteration 1, diff = -3.17e-1.
# Newton method iteration 2, diff = -7.78e-2.
# Newton method iteration 3, diff = -5.33e-3.
# Newton method iteration 4, diff = -2.52e-5.
# Newton method iteration 5, diff = -5.65e-10.
# Newton method iteration 6, diff = -2.83e-19.
# Newton method iteration 7, diff = -7.14e-38.
# Newton method iteration 8, diff = -4.53e-75.
# Newton method iteration 9, diff = -1.83e-149.
# ...
