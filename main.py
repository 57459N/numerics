import numpy as np
import random


def upper_triangle(matrix: np.array, rank: int):
    ...


def filled_matrix(rank: int):
    matrix = np.array(np.empty((rank, rank)))

    for row in range(rank):
        matrix[row] = [int(i) for i in input().split(' ')]

    return matrix


def main():
    # rank = int(input('Input rank of square matrix: '))
    rng = np.random.default_rng()

    matrix_a = rng.integers(-10, 10, size=(3, 3))
    matrix_b = rng.integers(10, size=(3, 1))
    matrix_ab = np.concatenate((matrix_a, matrix_b), axis=1, dtype=float)

    print(matrix_ab)

    first_col = matrix_a[:, 0]
    abs_max_pos = np.where(first_col == max(first_col.min(), first_col.max(), key=abs))[0][0]

    print(f'max abs row {abs_max_pos}')

    matrix_ab[[0, abs_max_pos]] = matrix_ab[[abs_max_pos, 0]]

    rank = matrix_ab.shape[0]
    for i in range(rank - 1):
        matrix_ab[i] /= matrix_ab[i][i]
        for j in range(i + 1, rank):
            matrix_ab[j] -= matrix_ab[i] * matrix_ab[j][i]
    matrix_ab[rank - 1] /= matrix_ab[rank - 1][rank - 1]

    matrix_solution = np.zeros(rank)

    for i in range(rank):
        current_line = rank - i - 1
        matrix_solution[current_line] = matrix_ab[current_line][rank] - sum(
            [matrix_solution[j] * matrix_ab[current_line][j] for j in range(rank)])

    print(matrix_solution)

    for i in range(rank):
        print((matrix_a[i] * matrix_solution).sum(), matrix_b[i])


if __name__ == "__main__":
    main()
