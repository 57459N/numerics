import numpy as np
import random


def upper_triangle(matrix: np.array):
    rank = matrix.shape[0]

    for i in range(rank - 1):
        matrix[i] /= matrix[i][i]
        for j in range(i + 1, rank):
            matrix[j] -= matrix[i] * matrix[j][i]
    matrix[rank - 1] /= matrix[rank - 1][rank - 1]

    return matrix


def solution_from_triangle(matrix: np.array):
    rank = matrix.shape[0]
    matrix_solution = np.zeros(rank)
    for i in range(rank):
        current_line = rank - i - 1
        matrix_solution[current_line] = matrix[current_line][rank] - sum(
            [matrix_solution[j] * matrix[current_line][j] for j in range(rank)])

    return matrix_solution


def solution(matrix: np.array):
    first_col = matrix[:, 0]
    abs_max_pos = np.where(first_col == max(first_col.min(), first_col.max(), key=abs))[0][0]
    matrix[[0, abs_max_pos]] = matrix[[abs_max_pos, 0]]

    matrix = upper_triangle(matrix)

    return solution_from_triangle(matrix)


def magnitude(vector):
    return np.sqrt(sum(i ** 2 for i in vector))


def relative_error(first_solution: np.array, second_solution: np.array):
    diff = second_solution - first_solution
    return np.abs(diff).max() / np.abs(first_solution).max()


def residual(matrix_ab: np.array, vector_solution: np.array):
    matrix_ax = np.array([(matrix_ab[i, :-1] * vector_solution).sum() for i in range(matrix_ab.shape[0])])
    vector_residual = matrix_ax - matrix_ab[:, -1]
    return vector_residual


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

    print('Begin matrix: ')
    print(matrix_ab)

    matrix_solution = solution(matrix_ab)

    print(f'\nSolution:\n {matrix_solution}')

    vector_residual = residual(matrix_ab, matrix_solution)

    print(f'\nResidual vector:\n {vector_residual}')

    delta_residual = np.abs(vector_residual).max()

    print(f'\nDelta:\n {delta_residual}')

    matrix_ax_tilde = np.dot(matrix_a, matrix_solution.reshape((3, 1)))
    matrix_ab_new = np.concatenate((matrix_a, matrix_ax_tilde), axis=1)

    print(f'\nMatrix Ax~:\n {matrix_ax_tilde}')
    print(f'\nNew matrix:\n {matrix_ab_new}')

    matrix_solution_new = solution(matrix_ab_new)

    print(f'\nSecond solution:\n {matrix_solution_new}')

    rel_error = relative_error(matrix_solution, matrix_solution_new)

    print(f'\nRelative error:\n {rel_error}')


if __name__ == "__main__":
    main()
