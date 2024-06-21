# 1. Testati ca un numar este prim

def is_prime(val):
    if val < 2 or val > 2 and val % 2 == 0:
        return False
    else:
        for i in range(3, val // 2 + 1):
            if val % i == 0:
                return False
    return True


li = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
result = []
for val in li:
    if (is_prime(val)):
        result.append(val)
print("1. " + str(result))

# 2. Ordonati cuvintele din fisier-ul  urmator: Latin-Lipsum.txt

try:
    with open('Latin-Lipsum.txt', 'r') as file:
        content = file.read().lower()
        content = sorted(content.split(' '))
    print("2. " + str(content))
except FileNotFoundError:
    print("The file does not exist!")
except IOError:
    print("An error occurs reading the file!")

'''
3. Data matricea si vectorul, afisati rezulatul produsului lor scalar.
[ 1  2  3  4
 11 12 13 14
 21 22 23 24 ] si vectorul [2, -5, 7, -10]
'''


def scalar_mul(matrix, vect):
    if len(vect) != len(matrix[0]):
        print("Err: no matching!")
    else:
        result = []
        for i in range(len(matrix)):
            current_result = 0
            for j in range(len(matrix[0])):
                current_result += matrix[i][j] * vect[j]
            result.append(current_result)
    return result


matrix = [[1, 2, 3, 4],
          [11, 12, 13, 14],
          [21, 22, 23, 24]]
vect = [2, -5, 7, -10]
print("3. " + str(scalar_mul(matrix, vect)))

import numpy as np
import math

'''
1. Creati matricea si vectorul de mai sus in numpy.
    a. Afisati doar ultimele 2 coloane din primele 2 randuri ale matricei
    b. Afisati ultimele 2 elemente din vector
'''

matrix = np.array([[1, 2, 3, 4],
                   [11, 12, 13, 14],
                   [21, 22, 23, 24]])
arr = np.array([2, -5, 7, -10])

print("\nI.\na. " + str(matrix[:2, -2:]) + ";\nb. " + str(arr[-2:]))

'''
2. Creati doi vectori cu numere aleatoare intre 0 si 1 de aceleasi dimensiuni. 
    a. Afisati care dintre cei doi vectori are suma elementelor mai are.
    b. Adunati cei doi vectori. Inmultiti cei doi vector (vectorial si scalar).
    c. Calculati radical din fiecare element din vector
'''
rng = np.random.default_rng()
arr1, arr2 = rng.integers(low=0, high=2, size=3), rng.integers(low=0, high=2, size=3)
print("II. " + f"arr1: {arr1}; arr2: {arr2}")
if arr1.sum() > arr2.sum():
    print(f"a.  sum({arr1})={arr1.sum()} > sum({arr2})={arr2.sum()}")
else:
    print(f"a.  sum({arr1})={arr1.sum()} < sum({arr2})={arr2.sum()}")
print(f"b.  sum: arr1 + arr2 = {arr1 + arr2}; "
      f"\n\tmul: arr1 * arr2 = {arr1 * arr2}; "
      f"\n\tscalar_mul (Dot Product): {np.dot(arr1, arr2)} or {arr1 @ arr2}"
      f"\n\tvectorial_mul (Cross Product): {np.cross(arr1, arr2)}")
print("c. ")
for val in arr1:
    print(math.sqrt(val))

'''
3. Creati o matrice cu numere aleatoare intre 0 si 1 de dimensiune 5x5. 
    a. Afisati transpusa matricei
    b. Afisati inversa matricei si determinantul.
'''
rng = np.random.default_rng()
matrix = rng.random((5, 5))
print("3. " + str(matrix))
print("a. Transposing:\n" + str(matrix.T))
print("a. inv and det:\n" + str(np.linalg.inv(matrix)) + "\n and \n" + str(np.linalg.det(matrix)))

'''
4. Creati un vector de dimensiune 5 cu numere aleatoare:
    a. Afisati produsul scalar intre matricea definite la punctul 3 si vectorul curent
'''

new_arr = rng.random(5)
print("4. doT Product: " + str(np.dot(new_arr, matrix)))

# useful link: https://numpy.org/devdocs/user/absolute_beginners.html#why-use-numpy
# tests using numpy:
print("\n====================")
print("\nTests using numpy:")

print("dim of arr: " + str(arr.ndim))
print("dim of matrix: " + str(matrix.ndim))

print("shape of arr: " + str(arr.shape))
print("shape of matrix: " + str(matrix.shape))

print("size of arr: " + str(arr.size))
print("size of matrix: " + str(matrix.size))

print("data_type of arr: " + str(arr.dtype))
print("data_type of matrix: " + str(matrix.dtype))

print("\nBasic arrays creation:")

print("zeros: " + str(np.zeros(5)))
print("ones:  " + str(np.ones(5)))
print("ones int:  " + str(np.ones(5, dtype=np.int64)))
print("empty: " + str(np.empty(5)))
print("range: " + str(np.arange(1, 10, 4)))
print("linespace: " + str(np.arange(1, 10, 2.5)))

print("\nsort, add, remove elems from arrs:")
arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
print("sorting: " + str(np.sort(arr)))
a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7])
print("concatenate: " + str(np.concatenate((a, b))))

print("\nreshape arrs:")
c = np.arange(6)
print("new arr: " + str(c))
print("reshape:\n" + str(c.reshape(2, 3)))
