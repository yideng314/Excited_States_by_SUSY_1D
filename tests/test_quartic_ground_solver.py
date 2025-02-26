import numpy as np
from decimal import Decimal, getcontext
import json
from src.riccati_equation_solver import newton_method
from src.quartic_ground_solver import *

if __name__ == "__main__":
    c = Decimal('4')
    E0 = Decimal('0.5301810452420914498235230083461845377837552542265523506989065424031814737369318062074841207671807295')
    truncation_order = 100
    precision = 100
    tol = Decimal('1e-95')
    E = quartic_ground_solver(c, E0, truncation_order, precision, tol=tol)
    print(E)
    
    filename = f"results/quartic_ground_state_c_{c}_truncation_order_{truncation_order}_precision_{precision}_tol_{tol}.json"
    with open(filename, 'r') as f:
        data = json.load(f)
    E = Decimal(data["energy"])
    expansion = np.array([Decimal(x) for x in data["expansion"]])
    print(E)
    print(expansion)

# Output:
# 
# Newton method iteration 1, diff = 1.47e-31.
# Newton method iteration 2, diff = -3.39e-54.
# Newton method iteration 3, diff = -1.88e-99.
# Newton method converges at iteration 3, with energy convergence -1.88e-99.
# 0.5301810452420914498235230083463317727576043642644856279971917456768829597476405690356032508158329702
# 0.5301810452420914498235230083463317727576043642644856279971917456768829597476405690356032508158329702
# ...
