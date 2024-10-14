#1. Parsing the System of Equations (1 point)
import pathlib
import re


def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A = []
    B = []
    with path.open('r') as file:
        lines = file.readlines()
    for line in lines:
        splitted_line = line.split()
        row = []
        sign = ''
        for i in splitted_line:
            if '+' in i or '-' in i:
                sign = i
            if 'x' in i or 'y' in i or 'z' in i:
                match = re.search(r'\d+', i)
                if match:
                    result = sign + match.group()
                    row.append(float(result))
                else:
                    result = sign + '1'
                    row.append(float(result))
            elif '+' not in i and '-' not in i and '=' not in i:
                B.append(i)
        A.append(row)
    return A, B


load_system(pathlib.Path("system.txt"))
A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=}, {B=}")


# Write a function to compute the determinant of matrix A.
def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 3:
        return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) - matrix[0][1] * (
                matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) + matrix[0][2] * (
                matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    else:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


print(f"{determinant(A)=}")


# Compute the sum of the elements along the main diagonal of matrix A.
def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


print(f"{trace(A)=}")


#Compute the Euclidean norm of vector B.
import math


def norm(vector: list[float]) -> float:
    return math.sqrt(pow(float(vector[0]), 2) + pow(float(vector[1]), 2) + pow(float(vector[2]), 2))


print(f"{norm(B)=}")


#Write a function to compute the transpose of matrix A.
def transpose(matrix: list[list[float]]) -> list[list[float]]:
    result_matrix = [row[:] for row in matrix]
    for i in range(len(result_matrix)):
        for j in range(len(result_matrix[0])):
            if i < j:
                result_matrix[i][j], result_matrix[j][i] = result_matrix[j][i], result_matrix[i][j]
    return result_matrix


print(f"{transpose(A)=}")
# print(f"{A=}")


# Write a function that multiplies matrix A with vector B.
def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = []
    for i in range(len(matrix)):
        summ = 0
        for j in range(len(matrix[0])):
            summ += matrix[i][j]*float(vector[j])
        result.append(summ)
    return result


print(f"{multiply(A, B)=}")


# Solving using Cramer's Rule
def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = []

    # def matrix_column_replace(m: list[list[float]], v: list[float], col) -> list[list[float]]:
    #     m_copy = [row[:] for row in m]
    #     for index in range(len(m)):
    #         m_copy[index][col] = float(v[index])
    #     return m

    for i in range(len(matrix[0])):
        new_matrix = [row[:] for row in matrix]
        for index in range(len(matrix)):
            new_matrix[index][i] = float(vector[index])
        value = determinant(new_matrix) / determinant(matrix)
        result.append(value)
    return result


print(f"{solve_cramer(A, B)=}")

# Solving using Inversion
def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    minor_list = []
    for row in range(len(matrix)):
        min_row = []
        for col in range(len(matrix[0])):
            if row != i and col != j:
                min_row.append(matrix[row][col])
        if min_row:
            minor_list.append(min_row)
    return minor_list

print(f"Minor computed: {minor(A, 0, 0)=}")

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    cofactor_matrix = matrix[:]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            cofactor_matrix[i][j] = (-1) ** (i+j) * determinant(minor(matrix, i, j))
    return cofactor_matrix

print(f"Cofactor matrix: {cofactor(A)}")

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))

print(f"Adjoint matrix: {adjoint(A)}")

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return multiply(adjoint(matrix),vector)

print(f"{solve(A, B)=}")