import numpy as np
import math
from decimal import Decimal, getcontext
from src.riccati_equation_solver import newton_method
import json
from src.quartic_excited_solver import *

if __name__ == "__main__":
    # set the parameters
    c = Decimal('4')
    E0s = [Decimal('0.53'), Decimal('1.90'), Decimal('3.72'), Decimal('5.82'), Decimal('8.13')]
    truncation_order = 10
    precision = 100
    tol = Decimal('1e-95')
    max_iter = 100
    n_excited = 5
    # solve the excited states
    Es, expansions = quartic_excited_solver(c, E0s, truncation_order, precision, tol, max_iter, n_excited)
    # print the results
    for i in range(len(Es)):
        print(f"Energy of the state {i}: {Es[i]}")
    # save the results to a JSON file
    filename = f"results/quartic_excited_{n_excited}_c_{c}_truncation_order_{truncation_order}_precision_{precision}_tol_{tol}.json"
    save_results(filename, Es, expansions)
    # load the results from the JSON file
    Es, expansions = load_results(filename)
    print(Es)

# Output:
# 
# Newton method iteration 1, diff = 1.82e-4.
# Newton method iteration 2, diff = -6.08e-7.
# Newton method iteration 3, diff = -6.84e-12.
# Newton method iteration 4, diff = -8.65e-22.
# Newton method iteration 5, diff = -1.38e-41.
# Newton method iteration 6, diff = -3.52e-81.
# Newton method iteration 7, diff = 1.33e-101.
# Newton method converges at iteration 7, with energy convergence 1.33e-101.
# Newton method iteration 1, diff = -1.63e-4.
# Newton method iteration 2, diff = -6.50e-7.
# Newton method iteration 3, diff = -1.04e-11.
# Newton method iteration 4, diff = -2.63e-21.
# Newton method iteration 5, diff = -1.69e-40.
# Newton method iteration 6, diff = -6.97e-79.
# Newton method iteration 7, diff = -2.85e-100.
# Newton method converges at iteration 7, with energy convergence -2.85e-100.
# Newton method iteration 1, diff = 1.02e-2.
# Newton method iteration 2, diff = -2.24e-3.
# Newton method iteration 3, diff = -1.36e-4.
# Newton method iteration 4, diff = -4.88e-7.
# Newton method iteration 5, diff = -6.28e-12.
# Newton method iteration 6, diff = -1.04e-21.
# Newton method iteration 7, diff = -2.85e-41.
# Newton method iteration 8, diff = -2.14e-80.
# Newton method iteration 9, diff = 2.28e-100.
# Newton method converges at iteration 9, with energy convergence 2.28e-100.
# Newton method iteration 1, diff = 2.55e-3.
# Newton method iteration 2, diff = -1.71e-4.
# Newton method iteration 3, diff = -8.03e-7.
# Newton method iteration 4, diff = -1.77e-11.
# Newton method iteration 5, diff = -8.59e-21.
# Newton method iteration 6, diff = -2.02e-39.
# Newton method iteration 7, diff = -1.12e-76.
# Newton method iteration 8, diff = -9.42e-102.
# Newton method converges at iteration 8, with energy convergence -9.42e-102.        
# Newton method iteration 1, diff = 9.39e-4.
# Newton method iteration 2, diff = -2.44e-5.
# Newton method iteration 3, diff = -1.67e-8.
# Newton method iteration 4, diff = -7.79e-15.
# Newton method iteration 5, diff = -1.70e-27.
# Newton method iteration 6, diff = -8.09e-53.
# Newton method iteration 7, diff = 3.97e-100.
# Newton method converges at iteration 7, with energy convergence 3.97e-100.
# Energy of the state 0: 0.5301810458149942644302911049487108471073878106995826127236307994113148938752843893812042040705062277
# Energy of the state 1: 1.899836529365607068532176443487812035808781487566415642901228785825238277645794975191073514834023804
# Energy of the state 2: 3.727849084156123431834677076509966906248976044232535182063149405356669421805193353311777325131272596
# Energy of the state 3: 5.822373272395362550143548701742161898779525045705771925222210396311540094464900797274184031783061201
# Energy of the state 4: 8.130914563347808095159439002064830083809509831616613019435835517279467329995435063091652742861126267
# [Decimal('0.5301810458149942644302911049487108471073878106995826127236307994113148938752843893812042040705062277'), Decimal('1.899836529365607068532176443487812035808781487566415642901228785825238277645794975191073514834023804'), Decimal('3.727849084156123431834677076509966906248976044232535182063149405356669421805193353311777325131272596'), Decimal('5.822373272395362550143548701742161898779525045705771925222210396311540094464900797274184031783061201'), Decimal('8.130914563347808095159439002064830083809509831616613019435835517279467329995435063091652742861126267')]
