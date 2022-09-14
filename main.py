import numpy as np


def upper_triangle(matrix: np.array):
    rank = matrix.shape[0]

    for i in range(rank):
        col = matrix[i:, i]
        abs_max_pos = np.where(col == max(col.min(), col.max(), key=abs))[0][0]
        matrix[[i, abs_max_pos + i]] = matrix[[abs_max_pos + i, i]]

        matrix[i] /= matrix[i][i]

        if i == rank - 1:
            break
        for j in range(i + 1, rank):
            matrix[j] -= matrix[i] * matrix[j][i]

    return matrix


def solution_from_triangle(matrix: np.array):
    rank = matrix.shape[0]
    matrix_solution = np.zeros(rank)
    for i in range(rank):
        current_line = rank - i - 1
        matrix_solution[current_line] = matrix[current_line][rank] - sum(
            [matrix_solution[j] * matrix[current_line][j] for j in range(rank)])

    return matrix_solution


def magnitude(vector):
    return np.sqrt(sum(i ** 2 for i in vector))


def relative_error(first_solution: np.array, second_solution: np.array):
    diff = second_solution - first_solution
    return np.abs(diff).max() / np.abs(first_solution).max()


def residual(matrix_ab: np.array, vector_solution: np.array):
    matrix_ax = np.array([(matrix_ab[i, :-1] * vector_solution).sum() for i in range(matrix_ab.shape[0])])
    vector_residual = matrix_ax - matrix_ab[:, -1]
    return vector_residual


def filled_matrix(text: str = ''):
    print(text)
    n = int(input('N: '))
    m = int(input('M: '))
    matrix = np.array(np.empty((n, m)))

    for row in range(n):
        matrix[row] = [float(i) for i in input().split(' ')]

    return matrix


def main():
    if input('m - manual/ a - auto (default): ') == 'm':
        matrix_a = filled_matrix('Enter matrix A: ')
        matrix_b = filled_matrix('Enter matrix B: ')
    else:
        rng = np.random.default_rng()
        matrix_a = rng.integers(-10, 10, size=(3, 3))
        matrix_b = rng.integers(10, size=(3, 1))

    matrix_ab = np.concatenate((matrix_a, matrix_b), axis=1, dtype=float)

    print('Begin matrix: ')
    print(matrix_ab)

    matrix_triangle = upper_triangle(matrix_ab)
    matrix_solution = solution_from_triangle(matrix_triangle)

    print(f'\nUpper triangle:\n {matrix_triangle}')
    print(f'\nSolution:\n {matrix_solution}')

    vector_residual = residual(matrix_ab, matrix_solution)

    print(f'\nResidual vector:\n {vector_residual}')

    delta_residual = np.abs(vector_residual).max()

    print(f'\nDelta:\n {delta_residual}')

    matrix_ax_tilde = np.dot(matrix_a, matrix_solution.reshape((3, 1)))
    matrix_ab_new = np.concatenate((matrix_a, matrix_ax_tilde), axis=1)

    print(f'\nMatrix Ax~:\n {matrix_ax_tilde}')
    print(f'\nNew matrix:\n {matrix_ab_new}')

    matrix_triangle_new = upper_triangle(matrix_ab_new)
    matrix_solution_new = solution_from_triangle(matrix_triangle_new)

    print(f'\nSecond triangle:\n {matrix_triangle_new}')
    print(f'\nSecond solution:\n {matrix_solution_new}')

    rel_error = relative_error(matrix_solution, matrix_solution_new)

    print(f'\nRelative error:\n {rel_error}')


if __name__ == "__main__":
    main()
