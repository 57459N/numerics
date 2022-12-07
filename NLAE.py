import copy
from scipy.linalg import solve as matrix_solve
import numpy as np
from typing import Callable, Optional
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


def get_J(k: float, arr: np.array, *funcs: Callable):
    J = np.matrix([np.zeros(len(arr))] * len(arr), float)
    for f_idx, f in enumerate(funcs):
        for num_idx, num in enumerate(arr):
            values = copy.copy(arr)
            values[num_idx] += values[num_idx] * k
            J[f_idx, num_idx] = (f(*values) - f(*arr)) / (arr[num_idx] * k)

    return J


def non_linear_solve(x0: np.array, e0: float, e1: float, derivative_k: float, *funcs: Callable, max_iters: int):
    iteration = 0

    delta1 = max([f(*x0) for f in funcs])
    delta2 = 1

    while (delta1 > e0 or delta2 > e1) and iteration < max_iters:
        iteration += 1
        yield x0, iteration

        F = np.matrix([f(*x0) for f in funcs]).T
        J = get_J(derivative_k, x0, *funcs)

        solution = matrix_solve(J, -F).T[0]
        x0 += solution

        delta1 = abs(max(np.asarray(F.T)[0], key=abs))
        delta2 = abs(max([dx if abs(dx) < 1 else dx / x0[idx] for idx, dx in enumerate(solution)], key=abs))


def non_linear_solve_manual(x, y, e0, e1, max_iters):
    iteration = 0

    delta1 = max(func1(x, y), func2(x, y))
    delta2 = 1

    while (delta1 > e0 or delta2 > e1) and iteration < max_iters:
        iteration += 1
        print(f'{iteration}:\tx: {x}\ty: {y}')

        F = np.matrix([func1(x, y), func2(x, y)]).T
        J = np.matrix([[func1_derivative_x1(x, y), func1_derivative_x2(x, y)],
                       [func2_derivative_x1(x, y), func2_derivative_x2(x, y)]])

        solution = matrix_solve(J, -F).T[0]
        x += solution[0]
        y += solution[1]

        delta1 = abs(max(np.asarray(F.T)[0], key=abs))
        delta2 = max(solution[0] if abs(solution[0]) < 1 else solution[0] / x,
                     solution[1] if abs(solution[1]) < 1 else solution[1] / y)


def many_derivatives_cases(koefs: list[float], x0: np.array, e0: float, e1: float, *funcs: Callable, max_iters: int):
    for i in koefs:
        print(f'\n{i} derivative K:\n')
        gen = non_linear_solve(x0, e0, e1, i, *funcs, max_iters=max_iters)
        while True:
            try:
                x, iters = next(gen)
                print(f'\t{iters}: ' + ' '.join([str(i) for i in x]))
            except StopIteration:
                break


def main():
    x0_1 = np.array([3.0, 2.0])
    x0_2 = np.array([3.0, -2.0])

    e0 = pow(10, -9)
    e1 = pow(10, -9)
    max_iters = 100
    get_J(0.1, x0_2, func1, func2)

    koefs = [0.1, 0.05, 0.01]

    many_derivatives_cases(koefs, x0_1, e0, e1, func1, func2, max_iters=max_iters)
    many_derivatives_cases(koefs, x0_2, e0, e1, func1, func2, max_iters=max_iters)

    print('\nmanual:\n')
    non_linear_solve_manual(x0_1[0], x0_1[1], e0, e1, max_iters)
    non_linear_solve_manual(x0_2[0], x0_2[1], e0, e1, max_iters)
    a = [-1, 0]


if __name__ == "__main__":
    main()
