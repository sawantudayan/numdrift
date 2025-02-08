"""
Unit tests for the mdarray class in the Numdrift library.

This module contains test cases to verify the correctness of the mdarray
implementation, including creation, indexing, slicing, setting values,
and conversion to NumPy arrays.
"""

import unittest
import numpy as np
from numdrift.core import mdarray, MISSING_VALUE


class TestMDArray(unittest.TestCase):
    """
    Test suite for the mdarray class.
    """

    def test_array_creation(self):
        """
        Test whether mdarray correctly initializes data and handles missing values.
        """
        arr = mdarray([1, 2, None, 4, 5])
        self.assertEqual(arr.data.tolist(), [1, 2, 0, 4, 5])
        self.assertTrue(arr.mask[2])


    def test_indexing(self):
        """
        Test indexing in mdarray, ensuring correct retrieval of values,
        including missing values.
        """
        arr = mdarray([1, 2, None, 4, 5])
        self.assertEqual(arr[1], 2)
        self.assertIsNone(arr[2])


    def test_slicing(self):
        """
        Test slicing functionality in mdarray, ensuring that missing value
        masks are preserved.
        """
        arr = mdarray([1, 2, None, 4, 5])
        sliced = arr.slice(1, 4)
        self.assertEqual(sliced.data.tolist(), [2, 0, 4])
        self.assertTrue(sliced.mask[1])


    def test_setitem(self):
        """
        Test setting values in mdarray, including handling of missing values.
        """
        arr = mdarray([1, 2, None, 4, 5])
        arr[3] = None
        self.assertTrue(arr.mask[3])


    def test_numpy_conversion(self):
        """
        Test conversion of mdarray to a NumPy array, ensuring that missing
        values are correctly replaced with NaN.
        """
        arr = mdarray([1, 2, None, 4, 5])
        np_arr = arr.to_numpy()
        self.assertTrue(np.isnan(np_arr[2]))


if __name__ == '__main__':
    unittest.main()
