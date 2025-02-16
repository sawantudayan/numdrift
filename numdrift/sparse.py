import numpy as np
import numba as nb
from numdrift.core import MISSING_VALUE  # Import global missing value constant


class SparseMDArray:
    """
    A sparse array implementation for handling large arrays efficiently.
    Stores only nonzero and non-missing values.
    """

    def __init__(self, data):
        """
        Initialize a sparse array from a dense NumPy array or list.

        Parameters:
        - data: list, NumPy array, or mdarray

        Stores:
        - values: Nonzero and non-missing values
        - indices: Indices of nonzero values
        - mask: Boolean mask for missing values
        - shape: Shape of the original array
        """
        if isinstance(data, list):
            data = np.array(data)  # Convert list to NumPy array

        if len(data.shape) > 1:
            raise ValueError("SparseMDArray currently supports only 1D arrays.")

        self.shape = data.shape
        flat_data = data.flatten()

        # Identify nonzero and non-missing values
        nonzero_indices = np.where((flat_data != 0) & (flat_data != MISSING_VALUE))[0]
        self.values = flat_data[nonzero_indices]
        self.indices = nonzero_indices

        # Mask for missing values
        self.mask = np.full(data.shape, False, dtype=bool)
        self.mask[np.where(flat_data == MISSING_VALUE)] = True

    def to_dense(self):
        """
        Convert sparse representation back to a full NumPy array.
        Missing values are restored as `MISSING_VALUE`.
        """
        dense = np.zeros(self.shape, dtype=float)  # Default to zero
        dense[self.indices] = self.values  # Restore nonzero values
        dense[self.mask] = MISSING_VALUE  # Restore missing values
        return dense

    def __getitem__(self, index):
        """
        Retrieve element at the given index.

        - Returns `None` if the value is missing.
        - Returns `0` if the value is not stored.
        """
        if isinstance(index, tuple):  # Support multi-dimensional indexing
            index = np.ravel_multi_index(index, self.shape)

        if self.mask[index]:
            return None  # Return `None` for missing values

        if index in self.indices:
            return self.values[np.where(self.indices == index)[0][0]]

        return 0  # Default to zero

    def __setitem__(self, index, value):
        """
        Set an element at the given index while preserving sparsity.

        - If value is `None`, it is marked as missing.
        - If value is `0`, it is removed from the sparse storage.
        """
        if isinstance(index, tuple):  # Handle multi-dimensional indexing
            index = np.ravel_multi_index(index, self.shape)

        if value is None:
            self.mask[index] = True
            return

        self.mask[index] = False  # If setting value, it's not missing

        if value != 0:
            if index in self.indices:
                self.values[np.where(self.indices == index)[0][0]] = value
            else:
                self.indices = np.append(self.indices, index)
                self.values = np.append(self.values, value)
        else:
            if index in self.indices:
                idx = np.where(self.indices == index)[0][0]
                self.indices = np.delete(self.indices, idx)
                self.values = np.delete(self.values, idx)

    def add(self, other):
        """
        Element-wise addition with another sparse array.
        Returns a new `SparseMDArray`.
        """
        if self.shape != other.shape:
            raise ValueError("Shapes must match for element-wise addition.")

        result_dense = self.to_dense() + other.to_dense()
        return SparseMDArray(result_dense)

    def multiply(self, other):
        """
        Element-wise multiplication with another sparse array.
        Returns a new `SparseMDArray`.
        """
        if self.shape != other.shape:
            raise ValueError("Shapes must match for element-wise multiplication.")

        result_dense = self.to_dense() * other.to_dense()
        return SparseMDArray(result_dense)

    def to_mdarray(self):
        """
        Convert `SparseMDArray` back to an `mdarray`.
        """
        from numdrift.core import mdarray  # Avoid circular imports
        return mdarray(self.to_dense())

    def __repr__(self):
        return f"SparseMDArray(shape={self.shape}, nonzero_elements={len(self.values)})"


# ---------------------------------------------------------------
# New CSR Format for Efficient 2D Sparse Matrices
# ---------------------------------------------------------------

class CSRMatrix:
    """
    A Compressed Sparse Row (CSR) matrix implementation for efficient storage and computation.
    """

    def __init__(self, data, shape):
        """
        Initialize the CSR matrix.

        Parameters:
        - data: dict {(row, col): value} - Dictionary containing nonzero values.
        - shape: tuple (rows, cols) - Shape of the matrix.
        """
        self.shape = shape
        self.data, self.indices, self.indptr = self._convert_to_csr(data, shape)

    @staticmethod
    def _convert_to_csr(data, shape):
        """
        Convert a dictionary representation to CSR format.

        Parameters:
        - data: dict {(row, col): value} - Nonzero values in the matrix.
        - shape: tuple (rows, cols) - Dimensions of the matrix.

        Returns:
        - data: np.array - Nonzero values.
        - indices: np.array - Column indices of nonzero values.
        - indptr: np.array - Row pointers.
        """
        rows, cols = shape
        sorted_items = sorted(data.items())  # Ensure row-wise order

        values, indices, indptr = [], [], [0]
        current_row = 0
        count = 0

        for (row, col), value in sorted_items:
            while current_row < row:
                indptr.append(count)
                current_row += 1
            values.append(value)
            indices.append(col)
            count += 1

        # Finalize indptr for remaining rows
        for _ in range(current_row, rows):
            indptr.append(count)

        return np.array(values, dtype=np.float64), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)

    def to_dense(self):
        """Convert the sparse CSR matrix to a dense NumPy array."""
        dense = np.zeros(self.shape, dtype=np.float64)
        for row in range(self.shape[0]):
            for i in range(self.indptr[row], self.indptr[row + 1]):
                dense[row, self.indices[i]] = self.data[i]
        return dense

    def matvec(self, vector):
        """Multiply the CSR matrix with a dense vector."""
        if len(vector) != self.shape[1]:
            raise ValueError("Vector size must match the number of columns.")
        return _csr_matvec(self.data, self.indices, self.indptr, vector)

    def __repr__(self):
        return f"CSRMatrix(shape={self.shape}, nonzeros={len(self.data)})"


@nb.njit
def _csr_matvec(data, indices, indptr, vector):
    """Optimized matrix-vector multiplication for CSR format using Numba."""
    result = np.zeros(len(indptr) - 1)
    for row in range(len(result)):
        for i in range(indptr[row], indptr[row + 1]):
            result[row] += data[i] * vector[indices[i]]
    return result