# Module: src.quartic_excited_solver_hpc_cli
# Author: Yi Deng <dengyi@stu.pku.edu.cn>

import argparse
from decimal import Decimal
from src.quartic_excited_solver_hpc import seq_quartic_excited_solver

def main():
    parser = argparse.ArgumentParser(description="Solve the excited states of the quartic potential.")
    parser.add_argument('--c', type=str, required=True, help='The parameter of the transformation (as a string to handle Decimal conversion).')
    parser.add_argument('--E0s', type=str, nargs='+', required=True, help='List of initial energy guesses for the first n excited states (as strings to handle Decimal conversion).')
    parser.add_argument("--truncations", type=int, required=True, help="The maximum truncation order for the expansion.")
    parser.add_argument('--precision', type=int, required=True, help='The precision for the high precision arithmetic.')
    parser.add_argument('--tol', type=str, required=True, help='The tolerance for energy eigenvalue convergence (as a string to handle Decimal conversion).')
    parser.add_argument('--max_iter', type=int, default=100, help='The maximum number of iterations for the Newton method.')
    parser.add_argument('--n_excited', type=int, default=5, help='The number of excited states to solve.')

    args = parser.parse_args()

    c = Decimal(args.c)
    E0s = [Decimal(E0) for E0 in args.E0s]
    truncations = args.truncations
    precision = args.precision
    tol = Decimal(args.tol)
    max_iter = args.max_iter
    n_excited = args.n_excited

    seq_quartic_excited_solver(c, E0s, truncations, precision, tol, max_iter, n_excited)

if __name__ == "__main__":
    main()
