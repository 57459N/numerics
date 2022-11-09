from scipy.linalg import solve as matrix_solve
import numpy as np
from typing import Callable
from math import log10, log


def func1(x1: float, x2: float):
    return 2 * x1 ** 2 - x1 * x2 - 5 * x1 + 1


def func1_derivative_x1(x1: float, x2: float):
    return 4 * x1 - x2 - 5


def func1_derivative_x2(x1: float, x2: float):
    return x1


def func2(x1: float, x2: float):
    return x1 + 3 * log10(x1) - x2 ** 2


def func2_derivative_x1(x1: float, x2: float):
    return 1. - 3. / (x1 * log(10))


def func2_derivative_x2(x1: float, x2: float):
    return -2 * x2


def non_linear_solve(x1_begin, x2_begin, e0: float, e1: float, func1: Callable, func2: Callable, max_iters: int):
    iteration = 0

    delta1 = max(func1(x1_begin, x2_begin), func2(x1_begin, x2_begin))
    delta2 = 1

    x1 = x1_begin
    x2 = x2_begin

    while True:
        iteration += 1
        print(f'{iteration}:\tx1: {x1}\tx2:{x2}')

        F = np.matrix([[func1(x1, x2), func2(x1, x2)]]).T
        J_ondef = np.matrix([[func1_derivative_x1(x1, x2), func1_derivative_x2(x2, x2)],
                             [func2_derivative_x1(x1, x2), func2_derivative_x2(x1, x2)]])
        J_005 = np.matrix([[(func1(x1 + x1 * 0.05, x2) - func1(x1, x2)) / x1 * 0.05,
                            (func2(x1, x2 + x2 * 0.05) - func1(x1, x2)) / x2 * 0.05],
                           [(func2(x1 + x1 * 0.05, x2) - func2(x1, x2)) / x1 * 0.05,
                            (func2(x1, x2 + x2 * 0.05) - func2(x1, x2)) / x2 * 0.05]])

        J_001 = np.matrix([[(func1(x1 + x1 * 0.01, x2) - func1(x1, x2)) / x1 * 0.01,
                            (func2(x1, x2 + x2 * 0.01) - func1(x1, x2)) / x2 * 0.01],
                           [(func2(x1 + x1 * 0.01, x2) - func2(x1, x2)) / x1 * 0.01,
                            (func2(x1, x2 + x2 * 0.01) - func2(x1, x2)) / x2 * 0.01]])

        J_01 = np.matrix([[(func1(x1 + x1 * 0.1, x2) - func1(x1, x2)) / x1 * 0.1,
                           (func2(x1, x2 + x2 * 0.1) - func1(x1, x2)) / x2 * 0.1],
                          [(func2(x1 + x1 * 0.1, x2) - func2(x1, x2)) / x1 * 0.1,
                           (func2(x1, x2 + x2 * 0.1) - func2(x1, x2)) / x2 * 0.1]])

        dx_ondef = matrix_solve(J_ondef, F)
        dx_005 = matrix_solve(J_005, F)
        dx_001 = matrix_solve(J_001, F)
        dx_01 = matrix_solve(J_01, F)

        


def main():
    x0_1 = np.array([3.0, 2.0])
    x0_2 = np.array([3.0, -2.0])

    e0 = pow(10, -9)
    e1 = pow(10, -9)
    max_iters = 10000
    get_J(0.1, x0_2, func1, func2)
    # non_linear_solve(x0_1[0], x0_1[0], e0, e1, func1, func2, max_iters)


if __name__ == "__main__":
    main()
