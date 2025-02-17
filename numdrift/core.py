import numpy as np
import numba as nb
from numba import float64, boolean
from numba import njit


MISSING_VALUE = np.nan

# Suppress division by zero warnings
np.seterr(divide='ignore', invalid='ignore')


class mdarray:
    """
    A NumPy-like array with native missing data support and optimized operations.
    """

    def __init__(self, data, dtype=None, missing_value=MISSING_VALUE):
        """
        Initialize the mdarray with custom missing data handling.

        Parameters:
        - data: list, tuple, or NumPy array
        - dtype: optional, specifies the data type
        - missing_value: placeholder for missing values (default: custom object)
        """
        self.data = np.array(data)  # Store the data
        self.mask = np.isnan(self.data)  # Mask for missing values
        self.missing_value = missing_value

    def _infer_dtype(self, data):
        """Infer the data type, ensuring missing values are handled correctly."""
        if any(x is self.missing_value for x in data):
            return object  # Use object type for mixed data
        return type(data[0])

    def __repr__(self):
        """String representation of mdarray."""
        return f"mdarray(data={self.data}, mask={self.mask})"

    def __getitem__(self, index):
        """Retrieve elements with missing value support."""
        if isinstance(index, int):
            return None if self.mask[index] else self.data[index]
        return mdarray(self.data[index], dtype=self.data.dtype, missing_value=self.missing_value)


    def __setitem__(self, index, value):
        """Assign values while handling missing data."""
        if value is None or value is self.missing_value:
            self.mask[index] = True
            self.data[index] = 0  # Placeholder value
        else:
            self.mask[index] = False
            self.data[index] = value

    def slice(self, start, stop, step=1):
        """Slice the mdarray, preserving missing values."""
        return mdarray(self.data[start:stop:step], missing_value=self.missing_value)

    def to_numpy(self):
        """Convert mdarray to a NumPy array, replacing missing values with NaN."""
        arr = self.data.copy()
        arr[self.mask] = np.nan
        return arr

# ------------------------- Numba-optimized Helper Functions -------------------------
    @staticmethod
    @nb.njit
    def _elementwise_add(arr1, arr2, mask1, mask2):
        """Perform element-wise addition."""
        result = np.empty_like(arr1, dtype=np.float64)  # Ensure using correct dtype
        for i in range(len(arr1)):
            if mask1[i] or mask2[i]:
                result[i] = np.nan  # Explicitly handle the NaN case
            else:
                result[i] = arr1[i] + arr2[i]
        return result
    
    @staticmethod
    @nb.njit
    def _elementwise_multiply(arr1, arr2, mask1, mask2):
        """Perform element-wise multiplication."""
        result = np.empty_like(arr1, dtype=np.float64)
        for i in range(len(arr1)):
            if mask1[i] or mask2[i]:
                result[i] = np.nan
            else:
                result[i] = arr1[i] * arr2[i]
        return result

    def _apply_func(self, data, mask, func):
        """Apply the function element-wise while considering the mask for missing values."""
        result = np.empty_like(data, dtype=np.float64)  # Ensure correct dtype
        for i in range(len(data)):
            if mask[i]:
                result[i] = np.nan  # Handle missing values
            else:
                result[i] = func(data[i])  # Apply the function element-wise
        return result
    
    
# ------------------------- Optimized Element-wise Operations -------------------------
    
    def add(self, other):
        """Element-wise addition with missing data handling."""
        result_data = self._elementwise_add(self.data, other.data, self.mask, other.mask)
        new_mask = self.mask | other.mask
        return mdarray(result_data, missing_value=self.missing_value)

    def multiply(self, other):
        """Element-wise multiplication with missing data handling."""
        result_data = self._elementwise_multiply(self.data, other.data, self.mask, other.mask)
        new_mask = self.mask | other.mask
        return mdarray(result_data, missing_value=self.missing_value)

# ------------------------- Mathematical Functions -------------------------

    def apply_function(self, func):
        """
        Apply a mathematical function element-wise while preserving missing values.

        Parameters:
        - func: Function to apply (e.g., np.log, np.exp).

        Returns:
        - mdarray with the function applied.
        """
        result_data = self._apply_func(self.data, self.mask, func)
        return mdarray(result_data, missing_value=self.missing_value)

    def exp(self):
        """Compute the exponential function element-wise."""
        return self.apply_function(np.exp)

    def exp(self):
        """Compute the exponential function element-wise."""
        return self.apply_function(np.exp)

    def sin(self):
        """Compute the sine function element-wise."""
        return self.apply_function(np.sin)

    def cos(self):
        """Compute the cosine function element-wise."""
        return self.apply_function(np.cos)
    
    def log(self):
        """Compute the logarithm function element-wise."""
        return self.apply_function(np.log)

# ------------------------- Element-wise Arithmetic Operations -------------------------

    def __add__(self, other):
        """Element-wise addition while handling missing values."""
        new_mask = self.mask | other.mask
        result_data = np.where(new_mask, np.nan, self.data + other.data)
        return mdarray(result_data, missing_value=self.missing_value)

    def __sub__(self, other):
        """Element-wise subtraction while handling missing values."""
        new_mask = self.mask | other.mask
        result_data = np.where(new_mask, np.nan, self.data - other.data)
        return mdarray(result_data, missing_value=self.missing_value)

    def __mul__(self, other):
        """Element-wise multiplication while handling missing values."""
        new_mask = self.mask | other.mask
        result_data = np.where(new_mask, np.nan, self.data * other.data)
        return mdarray(result_data, missing_value=self.missing_value)

    def __truediv__(self, other):
        """Element-wise division while handling missing values and division by zero."""
        new_mask = self.mask | other.mask | (other.data == 0)  # Ensure no division by zero
        safe_div = np.where(other.data == 0, np.nan, self.data / other.data)
        result_data = np.where(new_mask, np.nan, safe_div)
        return mdarray(result_data, missing_value=self.missing_value)
    
    
    def _elementwise_op(self, other, op):
        """Generalized element-wise operation with missing value handling."""
        result_data = op(self.data, other.data)
        result_data[self.mask | other.mask] = np.nan
        return mdarray(result_data, missing_value=self.missing_value)


    # ------------------------- Statistical Operations -------------------------

    def mean(self):
        """Compute the mean, ignoring missing values."""
        valid_data = self.data[~self.mask]
        return np.nan if valid_data.size == 0 else np.mean(valid_data)

    def std(self):
        """Compute the standard deviation, ignoring missing values."""
        valid_data = self.data[~self.mask]
        return np.nan if valid_data.size == 0 else np.std(valid_data)

    def min(self):
        """Compute the minimum value, ignoring missing values."""
        valid_data = self.data[~self.mask]
        return np.nan if valid_data.size == 0 else np.min(valid_data)

    def max(self):
        """Compute the maximum value, ignoring missing values."""
        valid_data = self.data[~self.mask]
        return np.nan if valid_data.size == 0 else np.max(valid_data)

    def median(self):
        """Compute the median, ignoring missing values."""
        valid_data = self.data[~self.mask]
        return np.nan if valid_data.size == 0 else np.median(valid_data)

    def variance(self):
        """Compute the variance, ignoring missing values."""
        valid_data = self.data[~self.mask]
        return np.nan if valid_data.size == 0 else np.var(valid_data)

    def percentile(self, q):
        """
        Compute the q-th percentile, ignoring missing values.

        Parameters:
        - q: float, percentile value (0-100).

        Returns:
        - The q-th percentile of the array.
        """
        valid_data = self.data[~self.mask]
        return np.nan if valid_data.size == 0 else np.percentile(valid_data, q)