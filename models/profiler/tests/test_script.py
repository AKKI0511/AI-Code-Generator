import time
import random
import numpy as np

def generate_large_list():
    """Inefficiently generates a large list using append."""
    data = []
    for i in range(1_000_000):
        data.append(i ** 2)
    return data

def sum_large_list():
    """Processes the large list inefficiently."""
    data = generate_large_list()
    total = 0
    for num in data:
        total += num
    return total

def slow_matrix_multiplication():
    """Performs matrix multiplication inefficiently using nested loops."""
    size = 300
    matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
    matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]
    result = [[0] * size for _ in range(size)]

    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result

def fast_matrix_multiplication():
    """Performs matrix multiplication using NumPy."""
    size = 300
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    return np.dot(matrix_a, matrix_b)

def main():
    print("Profiling Test Script")

    start_time = time.time()
    print("Summing a large list...")
    total = sum_large_list()
    print(f"Sum result: {total}, Time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print("Performing slow matrix multiplication...")
    result = slow_matrix_multiplication()
    print(f"Matrix multiplication done, Time taken: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print("Performing fast matrix multiplication with NumPy...")
    result = fast_matrix_multiplication()
    print(f"NumPy matrix multiplication done, Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()