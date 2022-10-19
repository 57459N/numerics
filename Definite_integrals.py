from math import sqrt, fabs
from typing import Callable
from pprint import pprint
from numpy import arange


def trapezium_step(func: Callable, l_border: float, r_border: float, step: float):
    s = 0
    for i in arange(l_border + step, r_border, step, float):
        s += func(i)

    return step / 2 * (func(l_border) + 2 * s + func(r_border))


def trapezium_integral(func: Callable, l_border: float, r_border: float, precision: float) -> tuple[float, float]:
    step = .1
    solution_step = trapezium_step(func, l_border, r_border, step)
    solution_half_step = trapezium_step(func, l_border, r_border, step / 2)
    while fabs(solution_half_step - solution_step) >= precision ** 2:
        print(solution_half_step - solution_step)
        step /= 2
        solution_step = solution_half_step
        solution_half_step = trapezium_step(func, l_border, r_border, step / 2)
    print(solution_half_step - solution_step)

    return solution_half_step, step


def simpson_step(func: Callable, l_border: float, r_border: float, step: float):
    sum1 = 0
    sum2 = 0
    for i in arange(l_border + step, r_border, step / 2, float):
        sum1 += func(i)
    for i in arange(l_border + 2 * step, r_border - step, step / 2, float):
        sum2 += func(i)

    return step / 12 * (func(l_border) + 4 * sum1 + 2 * sum2 + func(r_border))


def simpson_integral(func: Callable, l_border: float, r_border: float, precision: float) -> tuple[float, float]:
    step = .1
    solution_step = simpson_step(func, l_border, r_border, step)
    solution_half_step = simpson_step(func, l_border, r_border, step / 2)
    while fabs(solution_half_step - solution_step) >= 15 * precision:
        print(solution_half_step - solution_step)
        step /= 2
        solution_step = solution_half_step
        solution_half_step = simpson_step(func, l_border, r_border, step / 2)
    print(solution_half_step - solution_step)

    return solution_half_step, step


def main():
    f = lambda x: sqrt(1 + 2 * x ** 3)
    l_border = 1.2
    r_border = 2.471
    precision = 10 ** (-5)
    solution_tr, final_step_tr = trapezium_integral(func=f, l_border=l_border, r_border=r_border, precision=precision)
    print(20*'-')
    solution_simp, final_step_simp = simpson_integral(func=f, l_border=l_border, r_border=r_border, precision=precision)
    print(f'Trapezium: {solution_tr}\nSimpson: {solution_simp}')


if __name__ == "__main__":
    main()
