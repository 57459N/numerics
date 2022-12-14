from typing import Callable, Optional
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from NLAE import non_linear_solve as n_solve


def du1dt_1(u1: float, u2: float, t: float, a: float):
    if t == 0:
        return -u1 * u2 + 1.

    return -u1 * u2 + np.sin(t) / t


def du2dt_1(u1: float, u2: float, t: float, a: float):
    return -(u2 * u2) + a * t / (1 + t * t)


def a_param_1(w: float):
    return 2.5 + w / 40.


def du1dt_2(u1: float, u2: float, t: float, a: float):
    return u2 - (a * u1 + 0.25 * u2) * u1


def du2dt_2(u1: float, u2: float, t: float, a: float):
    return np.exp(u1) - (u1 + a * u2) * u1


def explicit_euler_method(*funcs: Callable, param: Callable, param_values: np.ndarray, u0: np.array,
                          t_upper: float, E: float,
                          tau_max: float) -> None:
    for param_value in param_values:
        print(f'\t {param_value=}')

        t = 0
        y = u0

        iters = 0

        x_data = []
        y1_data = []
        y2_data = []

        while t < t_upper:
            f = np.array([f(*y, t=t, a=param(param_value)) for f in funcs], dtype=float)

            tau = min([E / (np.abs(f_i) + E / tau_max) for f_i in f])

            y = y + tau * f

            t = t + tau

            x_data.append(t)
            y1_data.append(y[0])
            y2_data.append(y[1])
            iters += 1

            print(f'{y=}\t{t=}')

        plt.title(f'Explicit method, param = {param_value}')
        plt.plot(x_data, y1_data)
        plt.plot(x_data, y2_data)
        plt.show()

        print(f'{iters=}')


def implicit_euler_method(*funcs: Callable, param: Callable, param_values: np.ndarray, u0: np.array,
                          t_upper: float, E: float, tau_min: float, tau_max: float):
    # 2
    for param_value in param_values:
        t = 0

        tau = tau_min
        tau_prev = tau_min

        y = u0
        y_next = u0
        y_prev = u0

        x_data = []
        y1_data = []
        y2_data = []

        while t < t_upper:
            while True:
                t_next = t + tau

                f = lambda y_, t_: np.array([f_(*y_, t=t_, a=param(param_value)) for f_ in funcs], dtype=float)

                y_next = y + tau * f(y, t)

                y_next = y + tau * f(y_next, t_next)

                Ek = np.max(np.abs(-1 * tau / (tau + tau_prev) * (y_next - y - tau / tau_prev * (y - y_prev))))

                if Ek <= E:
                    break

                tau /= 2
                t_next = t
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

            x_data.append(t)
            y1_data.append(y[0])
            y2_data.append(y[1])

            print(f'{y_next=} {t_next=}')

        plt.title(f'Implicit with clarification, param = {param_value}')
        plt.plot(x_data, y1_data)
        plt.plot(x_data, y2_data)
        plt.show()


def main():
    u0_1 = np.array([0., -0.412])
    u0_2 = np.array([1, 0.])

    t = 1
    E = 10 ** -1
    tau_max = 10 ** -2
    tau_min = 10 ** -3

    # gen = n_solve(u0, 3, 0.3, 0.01, du1dt, du2dt, max_iters=100, args=(0.1, a_param(25)))
    # while True:
    #     try:
    #         x, iters = next(gen)
    #         print(f'\t{iters}: ' + ' '.join([str(i) for i in x]))
    #     except StopIteration:
    #         break

    explicit_euler_method(du1dt_1, du2dt_1, param=a_param_1, param_values=[25.,], u0=u0_1, t_upper=t,
                          E=E, tau_max=tau_max)
    implicit_euler_method(du1dt_1, du2dt_1, param=a_param_1, param_values=np.array([25], dtype=float), u0=u0_1, t_upper=t,
                          E=E, tau_max=tau_max, tau_min=tau_min)

    explicit_euler_method(du1dt_2, du2dt_2, param=lambda x: 2, param_values=[1.], u0=u0_2,
                          t_upper=t, E=E, tau_max=tau_max)
    implicit_euler_method(du1dt_2, du2dt_2, param=lambda x: 2, param_values=np.array([1.], dtype=float), u0=u0_2,
                          t_upper=t, E=E, tau_max=tau_max, tau_min=tau_min)


if __name__ == "__main__":
    main()
