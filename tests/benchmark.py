"""
Benchmarking Script for Numdrift (mdarray)

This script benchmarks the performance of element-wise operations (addition
and multiplication) in the mdarray class using Numba optimizations.

- Generates two large random arrays.
- Converts them into mdarray objects.
- Measures execution time for addition and multiplication.
"""

import time
import numpy as np
from numdrift.core import mdarray

# Generate large test data
size = 1_000_000  # One million elements
data1 = np.random.rand(size)  # Random float values between 0 and 1
data2 = np.random.rand(size)

# Convert to mdarray
md1 = mdarray(data1)
md2 = mdarray(data2)


def benchmark_operation(operation, md1, md2):
    """
    Benchmarks a given mdarray operation and prints execution time.

    Parameters:
    - operation (str): The operation to benchmark ('add' or 'multiply').
    - md1 (mdarray): First input array.
    - md2 (mdarray): Second input array.

    Returns:
    - None (Prints the execution time)
    """
    start = time.time()

    if operation == "add":
        md1.add(md2)
    elif operation == "multiply":
        md1.multiply(md2)
    else:
        raise ValueError("Unsupported operation. Use 'add' or 'multiply'.")

    end = time.time()
    print(f"Numdrift {operation.capitalize()} Time: {end - start:.6f} seconds")


# Run benchmarks
benchmark_operation("add", md1, md2)
benchmark_operation("multiply", md1, md2)
