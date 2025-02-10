import numpy as np
import numba as nb

MISSING_VALUE = object()  # Unique object for missing values

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
        self.missing_value = missing_value
        self.data = np.array(data, dtype=dtype if dtype else self._infer_dtype(data))
        self.mask = np.array([x is missing_value for x in data], dtype=bool)


    def _infer_dtype(self, data):
        """Infer the data type, ensuring missing values are handled correctly."""
        if any(x is self.missing_value for x in data):
            return object  # Use object type for mixed data
        return type(data[0])


    def __repr__(self):
        """String representation of mdarray."""
        return f"mdarray(data={self.data}, mask={self.mask})"


    def __getitem__(self, index):
        """Retrieve elements with support for missing data."""
        if isinstance(index, int):
            return None if self.mask[index] else self.data[index]
        return mdarray(self.data[index], missing_value=self.missing_value)


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
        """Convert mdarray to a NumPy array, replacing missing values."""
        arr = self.data.copy()
        arr[self.mask] = np.nan
        return arr


    # Optimized Element-wise Operations using Numba
    def add(self, other):
        """Element-wise addition with missing data handling."""
        return mdarray(_elementwise_add(self.data, other.data, self.mask, other.mask), missing_value=self.missing_value)


    def multiply(self, other):
        """Element-wise multiplication with missing data handling."""
        return mdarray(_elementwise_multiply(self.data, other.data, self.mask, other.mask), missing_value=self.missing_value)


# Numba-optimized helper functions
@nb.njit
def _elementwise_add(arr1, arr2, mask1, mask2):
    """Numba-optimized element-wise addition, handling missing values."""
    result = np.empty_like(arr1, dtype=np.float64)
    for i in range(len(arr1)):
        if mask1[i] or mask2[i]:
            result[i] = np.nan  # Handle missing values
        else:
            result[i] = arr1[i] + arr2[i]
    return result


@nb.njit
def _elementwise_multiply(arr1, arr2, mask1, mask2):
    """Numba-optimized element-wise multiplication, handling missing values."""
    result = np.empty_like(arr1, dtype=np.float64)
    for i in range(len(arr1)):
        if mask1[i] or mask2[i]:
            result[i] = np.nan  # Handle missing values
        else:
            result[i] = arr1[i] * arr2[i]
    return result