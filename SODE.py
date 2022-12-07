from typing import Callable
from scipy.optimize import fsolve

import numpy as np
import matplotlib.pyplot as plt


def du1dt(u1: float, u2: float, t: float, a: float):
    if t == 0:
        return u1 * u2 + 1.

    return -u1 * u2 + np.sin(t) / t


def du2dt(u1: float, u2: float, t: float, a: float):
    return -(u2 * u2) + a * t / (1 + t * t)


def a_param(w: float):
    return 2.5 + w / 40.


def explicit_euler_method(*funcs: Callable, u0: np.array, t_upper: float, E: float, tau_max: float):
    for w in range(25, 48, 1):
        print(f'\t {w=}')

        t = 0
        y = u0

        iters = 0

        x_data = []
        y_data = []

        while (t < t_upper):
            f = np.array([f(*y, t=t, a=a_param(w)) for f in funcs], dtype=float)

            tau = min([E / (np.abs(f_i) + E / tau_max) for f_i in f])

            y = y + tau * f

            t = t + tau

            x_data.append(y[0])
            y_data.append(y[1])
            iters += 1

            print(f'{y=}\t{t=}')

        plt.title(f'{w=}')
        plt.plot(x_data, y_data)
        plt.show()

        print(f'{iters=}')


def equation(var, *data):
    t, a = data
    u1, u2 = var
    eq1 = du1dt(u1, u2, t, a)
    eq2 = du1dt(u1, u2, t, a)
    return [eq1, eq2]


def implicit_euler_method(*funcs: Callable, u0: np.array, t_upper: float, E: float, tau_min: float, tau_max: float):
    # 2
    t = 0

    tau = tau_min
    tau_prev = tau_min
    tau_next = tau_min

    y = u0
    y_next = u0
    y_prev = u0

    x_data = []
    y_data = []

    Ek = ...
    while t < t_upper:
        t_next = 0
        while True:
            t_next = t + tau

            y_next = fsolve(equation, y_next, args=(t_next, a_param(40)))

            Ek = np.max(np.abs(-1 * (tau / (tau + tau_prev) * (y_next - y - (tau / tau_prev * (y - y_prev))))))

            if Ek <= E:
                break

            tau /= 2
            tau_next = tau
            y_next = y
            print(1)

        tau_next = (E / Ek) ** 0.5 * tau

        if tau_next > tau_max:
            tau_next = tau_max

        y_prev = y
        y = y_next
        tau_prev = tau
        tau = tau_next
        t = t_next

        x_data.append(y[0])
        y_data.append(y[1])

        print(f'{y_next=} {t_next=}')

    plt.plot(x_data, y_data)
    plt.show()


def main():
    u0 = np.array([0., -0.412])
    t = 1
    E = 10 ** -6
    tau_max = 10 ** -5
    tau_min = 10 ** -8

    explicit_euler_method(du1dt, du2dt, u0=u0, t_upper=t, E=E, tau_max=tau_max)
    # implicit_euler_method(du1dt, du2dt, u0=u0, t_upper=t, E=E, tau_max=tau_max, tau_min=tau_min)


if __name__ == "__main__":
    main()
