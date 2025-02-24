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

    def __init__(self, data, dtype=None, missing_value=MISSING_VALUE, mask=None):
        """
        Initialize the mdarray with custom missing data handling.

        Parameters:
        - data: list, tuple, or NumPy array
        - dtype: optional, specifies the data type
        - missing_value: placeholder for missing values (default: custom object)
        """
        self.data = np.array(data)  # Store the data
        self.missing_value = missing_value
        
        if mask is not None:
            # Usinf provided mask
            self.mask = mask
        else:
            # Generate MASK based in MISSING_VALUE
            self.mask = np.isnan(self.data)
            
        # Applying MISSING_VALUE to masked positions
        self.data[self.mask] = np.nan

    def _infer_dtype(self, data):
        """Infer the data type, ensuring missing values are handled correctly."""
        if any(x is self.missing_value for x in data):
            return object  # Use object type for mixed data
        return type(data[0])

    def __repr__(self):
        """String representation of mdarray."""
        return f"mdarray(data={self.data}, mask={self.mask})"

    def __getitem__(self, index):
        """
        Advanced indexing for mdarray.
        
        Supports slicing, fancy indexing, and boolean masking.

        Parameters:
        - index: int, slice, list, ndarray, or boolean mask.

        Returns:
        - mdarray: Subset of the original mdarray based on the index.
        """
        if isinstance(index, tuple):
            # Multi-dimensional indexing
            data = self.data[index]
            mask = self.mask[index]
        elif isinstance(index, (int, slice)):
            # Simple slicing
            data = self.data[index]
            mask = self.mask[index]
        elif isinstance(index, (list, np.ndarray)):
            index = np.asarray(index)
            if index.dtype == bool:
                # Boolean indexing
                data = self.data[index]
                mask = self.mask[index]
            else:
                # Fancy indexing
                data = self.data[index]
                mask = self.mask[index]
        else:
            raise IndexError("Unsupported index type.")

        return mdarray(data, mask=mask)


    def __setitem__(self, index, value):
        """
        Set values in mdarray using advanced indexing.

        Parameters:
        - index: int, slice, list, ndarray, or boolean mask.
        - value: scalar or array-like to assign.
        """
        if isinstance(index, tuple):
            self.data[index] = value
            if np.isnan(value):
                self.mask[index] = True
            else:
                self.mask[index] = False
        elif isinstance(index, (int, slice)):
            self.data[index] = value
            self.mask[index] = np.isnan(value)
        elif isinstance(index, (list, np.ndarray)):
            index = np.asarray(index)
            if index.dtype == bool:
                self.data[index] = value
                self.mask[index] = np.isnan(value)
            else:
                self.data[index] = value
                self.mask[index] = np.isnan(value)
        else:
            raise IndexError("Unsupported index type.")


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
    
    
    # ------------------------- Broadcasting & Indexing Operations -------------------------
    
    def _prepare_operand(self, other):
        """
        Prepare the operand for element-wise operations, ensuring data and mask are aligned.

        Parameters:
        ----------
        other : mdarray, array-like, or scalar
            The operand to align for broadcasting.

        Returns:
        -------
        tuple of np.ndarray
            Aligned data and mask arrays for the operand.
	    """
        if isinstance(other, mdarray):
            return other.data, other.mask
        elif np.isscalar(other):
            # Convert scalar to an array and mask as False (no missing values)
            return np.full_like(self.data, other, dtype=float), np.full_like(self.mask, False)
        else:
            # Assume it's array-like (list or np.ndarray)
            other_arr = np.array(other, dtype=float)
            other_mask = np.isnan(other_arr)
            return other_arr, other_mask
    
    
    
    def _broadcast_to(self, shape):
        """
        Broadcast the mdarray to a new shape following NumPy broadcasting rules.

        Parameters:
        - shape: tuple - Target shape to broadcast to.

        Returns:
        - mdarray: A new mdarray broadcasted to the target shape.
        """
        broadcasted_data = np.broadcast_to(self.data, shape)
        broadcasted_mask = np.broadcast_to(self.mask, shape)
        broadcasted_array = mdarray(broadcasted_data)
        broadcasted_array.mask = broadcasted_mask
        return broadcasted_array
    
    
    
    def _apply_elementwise_op(self, other, op):
        """
        Apply an element-wise operation with broadcasting, handling missing values.

        Parameters:
        ----------
        other : mdarray, array-like, or scalar
            The array or scalar to operate with.
        op : function
            The NumPy function representing the operation (e.g., np.add, np.multiply).

        Returns:
        -------
        mdarray
            Resulting mdarray after applying the operation.
        """
        other_data, other_mask = self._prepare_operand(other)

        # Apply broadcasting
        broadcasted_self_data, broadcasted_other_data = np.broadcast_arrays(self.data, other_data)
        broadcasted_self_mask, broadcasted_other_mask = np.broadcast_arrays(self.mask, other_mask)

        # Perform operation with masking
        result_data = op(broadcasted_self_data, broadcasted_other_data).astype(float)  # Force float dtype
        result_mask = broadcasted_self_mask | broadcasted_other_mask  # Mask where either is missing

        # Assign np.nan to masked positions
        result_data[result_mask] = np.nan

        return mdarray(result_data, result_mask)
    
    
    # ------------------------- Arithmetic Operations using Broadcasting -------------------------
    
    def add(self, other):
        """Element-wise addition with broadcasting."""
        return self._apply_elementwise_op(other, np.add)


    def subtract(self, other):
        """Element-wise subtraction with broadcasting."""
        return self._apply_elementwise_op(other, np.subtract)


    def multiply(self, other):
        """Element-wise multiplication with broadcasting."""
        return self._apply_elementwise_op(other, np.multiply)


    def divide(self, other):
        """Element-wise division with broadcasting."""
        return self._apply_elementwise_op(other, np.divide)
    
    
    # ------------------------- Arithmetic Operations using Broadcasting -------------------------


        
