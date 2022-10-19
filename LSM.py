import numpy as np
import scipy
import matplotlib.pyplot as pyplot
from pprint import pprint


def my_parse(table: str, x_lable: str, y_lable: str):
    table = table.replace('\n', ' ')
    begin_list = table.split(' ')
    data_dict = {x_lable: [], y_lable: []}

    status = x_lable
    for item in begin_list:
        if item == x_lable:
            status = x_lable
        elif item == y_lable:
            status = y_lable
        elif item.replace('.', '', 1).isdigit():
            data_dict[status].append(float(item))
    return data_dict[x_lable], data_dict[y_lable]


def main():
    # print(scipy.linalg.solve(matrix_a, matrix_b))

    table = '''t
    0.0 5.0 10.0 15.0 20.0 25.0
    C 1.00762 1.00392 1.00153 1.00000 0.99907 0.99852
    t 30 35 40 45 50 55
    C 0.99826 0.99818 0.99828 0.99849 0.99878 0.99919
    t 60.0 65.0 70.0 75.0 80.0 85.0
    C 0.99967 1.00024 1.00091 1.00167 1.00253 1.00351
    t 90.0 95.0 100.0 - - -
    C 1.00461 1.00586 1.00721 - - -'''

    x_values, y_values = my_parse(table, 't', 'C')

    rank = 4
    powers = [[x ** k for x in x_values] for k in range(rank * 2 - 1)]
    powers_sums = [sum(a) for a in powers]
    matrix_a = [[s for s in powers_sums[line:line + rank]] for line in range(rank)]
    matrix_b = [[sum([x * y for x, y in zip(powers[k], y_values)])] for k in range(rank)]
    solution = scipy.linalg.solve(matrix_a, matrix_b)
    y_values_lsm = [sum([(x ** k) * solution[k, 0] for k in range(rank)]) for x in x_values]

    dispersion = sum([(y0 - y1) ** 2 for y0, y1 in zip(y_values, y_values_lsm)])/(len(x_values) - rank - 2)
    mean_quadratic_deviation = dispersion ** 0.5

    pprint(matrix_a)
    pprint(matrix_b)
    print(solution)
    print(y_values_lsm)
    print(dispersion)
    print(mean_quadratic_deviation)

    # _, plots = pyplot.subplots(2)
    pyplot.plot(x_values, y_values, 'o')
    pyplot.plot(x_values, y_values_lsm)

    pyplot.show()


if __name__ == "__main__":
    main()
