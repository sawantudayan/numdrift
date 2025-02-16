import numpy as np
import time
from scipy.sparse import csr_matrix as scipy_csr
from numdrift.sparse import CSRMatrix


def generate_random_sparse_matrix(shape, density=0.01, seed=None):
    """
    Generate a random sparse matrix with a given shape and density.

    Parameters:
    - shape: tuple (rows, cols) - Shape of the matrix.
    - density: float - Proportion of non-zero values (between 0 and 1).
    - seed: int - Random seed for reproducibility.

    Returns:
    - Sparse matrix in CSR format as a CSRMatrix object.
    """
    if seed is not None:
        np.random.seed(seed)

    rows, cols = shape
    total_elements = rows * cols
    non_zero_elements = int(total_elements * density)

    # Generate random indices and values
    row_indices = np.random.randint(0, rows, non_zero_elements)
    col_indices = np.random.randint(0, cols, non_zero_elements)
    values = np.random.rand(non_zero_elements)

    # Convert to CSR format using the custom CSRMatrix
    data = {(row, col): value for row, col, value in zip(row_indices, col_indices, values)}
    sparse_matrix = CSRMatrix(data, shape)

    return sparse_matrix


def benchmark_sparse_matrix_operations(matrix, vector, iterations=10):
    """
    Benchmark matrix-vector multiplication using custom CSR and scipy CSR.

    Parameters:
    - matrix: SparseMatrix object - The matrix to multiply.
    - vector: np.ndarray - The vector to multiply with.
    - iterations: int - Number of iterations for benchmarking.

    Returns:
    - Dictionary with time taken for custom CSR and scipy CSR.
    """
    # Benchmark custom CSR matrix multiplication
    start = time.time()
    for _ in range(iterations):
        matrix.matvec(vector)
    custom_csr_time = time.time() - start

    # Benchmark scipy CSR matrix multiplication
    scipy_matrix = scipy_csr(matrix.to_dense())  # Convert to dense first for scipy
    start = time.time()
    for _ in range(iterations):
        scipy_matrix.dot(vector)
    scipy_csr_time = time.time() - start

    return {
        'custom_csr_time': custom_csr_time / iterations,
        'scipy_csr_time': scipy_csr_time / iterations
    }


def convert_dense_to_sparse(data, shape, threshold=0):
    """
    Convert a dense NumPy array to a sparse CSR matrix. Values below a threshold are considered zero.

    Parameters:
    - data: np.ndarray - The dense matrix.
    - shape: tuple (rows, cols) - Shape of the matrix.
    - threshold: float - Threshold below which elements are treated as zero.

    Returns:
    - CSRMatrix object representing the sparse matrix.
    """
    non_zero_indices = np.where(np.abs(data) > threshold)
    data_dict = {(row, col): data[row, col] for row, col in zip(*non_zero_indices)}

    return CSRMatrix(data_dict, shape)


def handle_missing_values(data, missing_value=MISSING_VALUE):
    """
    Replace missing values in a dense NumPy array with a specified missing value constant.

    Parameters:
    - data: np.ndarray - The input data with potential missing values.
    - missing_value: scalar - The value to replace missing values with.

    Returns:
    - The array with missing values replaced.
    """
    data[np.isnan(data)] = missing_value
    return data


def sparse_to_dense(matrix):
    """
    Convert a sparse CSRMatrix to a dense numpy array.

    Parameters:
    - matrix: CSRMatrix - The sparse matrix to convert.

    Returns:
    - Dense numpy array equivalent of the sparse matrix.
    """
    return matrix.to_dense()


def random_sparse_matrix_benchmark(shape, density=0.01, vector_size=None, iterations=10):
    """
    Benchmark the random sparse matrix generation and matrix-vector multiplication.

    Parameters:
    - shape: tuple (rows, cols) - Shape of the matrix.
    - density: float - Density of non-zero values in the matrix.
    - vector_size: int - Size of the vector for multiplication.
    - iterations: int - Number of iterations for benchmarking.

    Returns:
    - Benchmark results with matrix generation and multiplication times.
    """
    # Generate a random sparse matrix
    sparse_matrix = generate_random_sparse_matrix(shape, density)
    vector = np.random.rand(shape[1]) if vector_size is None else np.random.rand(vector_size)

    # Benchmark the operations
    benchmark_results = benchmark_sparse_matrix_operations(sparse_matrix, vector, iterations)
    return benchmark_results