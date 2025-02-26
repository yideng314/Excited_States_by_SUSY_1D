import numpy as np
from decimal import Decimal, getcontext
import math
import json
from src.quartic_ground_solver import quartic_ground_solver
from src.quartic_ground_solver_hpc import *

if __name__ == "__main__":
    c = Decimal('2')
    E0 = Decimal('0.5')
    truncations = 20
    precision = 100
    tol = Decimal('1e-95')
    seq_quartic_ground_solver(c, E0, truncations, precision, tol)

# Output:
# 
# Newton method iteration 1, diff = -1.00e-1.
# Newton method iteration 2, diff = -2.35e-2.
# Newton method iteration 3, diff = -1.46e-3.
# Newton method iteration 4, diff = -5.72e-6.
# Newton method iteration 5, diff = -8.73e-11.
# Newton method iteration 6, diff = -2.03e-20.
# Newton method iteration 7, diff = -1.10e-39.
# Newton method iteration 8, diff = -3.24e-78.
# Newton method iteration 9, diff = -1.67e-100.
# Newton method converges at iteration 9, with energy convergence -1.67e-100.
# Newton method iteration 1, diff = 2.09e-1.
# Newton method iteration 2, diff = -5.41e-2.
# Newton method iteration 3, diff = -5.38e-3.
# Newton method iteration 4, diff = -5.09e-5.
# Newton method iteration 5, diff = -4.52e-9.
# Newton method iteration 6, diff = -3.57e-17.
# Newton method iteration 7, diff = -2.23e-33.
# Newton method iteration 8, diff = -8.71e-66.
# Newton method iteration 9, diff = -5.92e-101.
# Newton method converges at iteration 9, with energy convergence -5.92e-101.
# ...
