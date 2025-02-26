# Module: test_riccati_equation_solver
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import numpy as np
import math
from decimal import Decimal, getcontext
import json
from src.riccati_equation_solver import *

if __name__ == "__main__":
    # set the precision
    getcontext().prec = 100
    # set the truncation order
    c = Decimal(4)
    f1 = np.array([0, -2, 9, -16, 14, -6, 1])
    f2 = np.array([0, -2, 1])
    f3 = np.array([-1, 1, 4, -8, 5, -1])
    f4 = np.array([0, 0, 4, -4, 1])*c**3
    f5 = np.array([2, -8, 12, -8, 2])*c 
    # set the energy guess
    E0 = Decimal('0.53018')
    # truncation order
    truncation_order = 10
    precision = 100
    tol = 1e-50
    # solve the Riccati equation
    E, expansion = newton_method(f1, f2, f3, f4, f5, E0, truncation_order, precision=precision, max_iter=100, tol=tol)
    print(E)
    # save the energy eigenvalue and the taylor coefficients list of the solution to a JSON file
    # filename contains truncation_order, precision, and tol
    filename = f"results/energy_and_expansion_truncation_order_{truncation_order}_precision_{precision}_tol_{tol}.json"
    save_energy_and_expansion_to_json(filename, E, expansion)
    
    # load the energy eigenvalue and the taylor coefficients list of the solution from a JSON file
    E, expansion = load_energy_and_expansion_from_json(filename=filename)
    print(E)
    print(expansion)

# Output:
# 
# Newton method iteration 1, diff = 1.05e-6.
# Newton method iteration 2, diff = -2.02e-11.
# Newton method iteration 3, diff = -7.55e-21.
# Newton method iteration 4, diff = -1.05e-39.
# Newton method iteration 5, diff = -2.05e-77.
# Newton method converges at iteration 5, with energy convergence -2.05e-77.
# 0.5301810458149942644302911049487108471073878106995826127236307994113148938753048762513465344076095990
# 0.5301810458149942644302911049487108471073878106995826127236307994113148938753048762513465344076095990
# [Decimal('-4.241448366519954115442328839589686776859102485596660901789046395290519151002439010010772275260876792')
#  Decimal('-7.751807797383237233188360713737060427256239917224475708341024928156429896714062473910932429306790547')
#  Decimal('4.50835050553435470343624208042055811754062042138541655632583087787613474225955509988270846485584697')
#  Decimal('-0.6225408705691619853108928804930889536265377395793257411313012389419115055403433236460466260874624329')
#  Decimal('-0.06091962080674487868017293072960969913905957291387358336171596810788303672316130332676454359877334689')
#  Decimal('0.1541807598026442423169625345627732125291369745583905045615475198595747135186410292081712186837376373')
#  Decimal('0.03612788451907309825753599635629520782258488856172549388771981899841352598746543995631360884124318')
#  Decimal('-0.0208682411009301303639882830313537228116154167342433007083121214045426031038169224147953983986398864')
#  Decimal('-0.01248375896677375388661963451322076602552714308202465026974500758380587670634980403498335109965924588')
#  Decimal('-1.360903765698925427927999651280263157894736842105263157894736842105263157894736842105263157894736842E-70')]
