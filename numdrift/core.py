"""
Core module for the Numdrift library.

This module defines the `mdarray` class, which extends NumPy arrays
with native missing data support and enhanced indexing operations.
"""

import numpy as np

MISSING_VALUE = object()  # Unique object for missing values


class mdarray:
    """
    A custom multi-dimensional array class that extends NumPy arrays
    with native missing data support.

    Attributes:
        data (np.ndarray): The main data array.
        mask (np.ndarray): A boolean mask indicating missing values.
        missing_value (Any): A placeholder for missing values.
    """


    def __init__(self, data, dtype=None, missing_value=MISSING_VALUE):
        """
        Initialize the mdarray with custom missing data handling.

        Args:
            data (list, tuple, or np.ndarray): Input data.
            dtype (optional, type): Specifies the data type (inferred if None).
            missing_value (optional, Any): Placeholder for missing values
                                           (default: unique object).
        """
        self.missing_value = missing_value
        self.data = np.array(data, dtype=dtype if dtype else self._infer_dtype(data))
        self.mask = np.array([x is missing_value for x in data], dtype=bool)


    def _infer_dtype(self, data):
        """
        Infer the appropriate data type while considering missing values.

        Args:
            data (list or np.ndarray): The input data.

        Returns:
            dtype: The inferred data type.
        """
        if any(x is self.missing_value for x in data):
            return object  # Use object type for mixed data
        return type(data[0])


    def __repr__(self):
        """
        String representation of mdarray.

        Returns:
            str: A formatted string displaying the array and mask.
        """
        return f"mdarray(data={self.data}, mask={self.mask})"


    def __getitem__(self, index):
        """
        Retrieve elements from the mdarray, handling missing values.

        Args:
            index (int or slice): The index or slice to retrieve.

        Returns:
            mdarray or element: The requested element or a sliced mdarray.
        """
        if isinstance(index, int):
            return None if self.mask[index] else self.data[index]
        return mdarray(self.data[index], missing_value=self.missing_value)


    def __setitem__(self, index, value):
        """
        Assign values to the mdarray while handling missing data.

        Args:
            index (int or slice): The index where the value should be assigned.
            value (Any): The new value to assign.

        Raises:
            ValueError: If the provided value is incompatible with the dtype.
        """
        if value is None or value is self.missing_value:
            self.mask[index] = True
            self.data[index] = 0  # Placeholder value for missing data
        else:
            self.mask[index] = False
            self.data[index] = value


    def slice(self, start, stop, step=1):
        """
        Slice the mdarray, preserving missing values.

        Args:
            start (int): Start index of the slice.
            stop (int): Stop index of the slice.
            step (int, optional): Step size (default is 1).

        Returns:
            mdarray: A new mdarray containing the sliced data.
        """
        return mdarray(self.data[start:stop:step], missing_value=self.missing_value)


    def to_numpy(self):
        """
        Convert the mdarray to a NumPy array, replacing missing values with NaN.

        Returns:
            np.ndarray: A NumPy representation of the mdarray.
        """
        arr = self.data.copy()
        arr[self.mask] = np.nan
        return arr
