"""
Unit tests for the mdarray class in the Numdrift library.

This module contains test cases to verify the correctness of the mdarray
implementation, including creation, indexing, slicing, setting values,
arithmetic operations, function applications, and NumPy conversions.
"""

import unittest
import numpy as np
from numdrift.core import mdarray, MISSING_VALUE


class TestMDArray(unittest.TestCase):
    """
    Test suite for the mdarray class.
    """

    def test_array_creation(self):
        """Test mdarray initialization and missing value handling."""
        arr = mdarray([1, 2, MISSING_VALUE, 4, 5])
        self.assertEqual(arr.data.tolist(), [1, 2, 0, 4, 5])
        self.assertTrue(arr.mask[2])

    def test_indexing(self):
        """Test correct indexing behavior, including missing values."""
        arr = mdarray([1, 2, MISSING_VALUE, 4, 5])
        self.assertEqual(arr[1], 2)
        self.assertIsNone(arr[2])

    def test_slicing(self):
        """Test slicing functionality, ensuring missing value masks are preserved."""
        arr = mdarray([1, 2, MISSING_VALUE, 4, 5])
        sliced = arr.slice(1, 4)
        self.assertEqual(sliced.data.tolist(), [2, 0, 4])
        self.assertTrue(sliced.mask[1])

    def test_setitem(self):
        """Test setting values in mdarray, including handling of missing values."""
        arr = mdarray([1, 2, MISSING_VALUE, 4, 5])
        arr[3] = MISSING_VALUE
        self.assertTrue(arr.mask[3])

    def test_numpy_conversion(self):
        """Test conversion of mdarray to NumPy array, ensuring missing values become NaN."""
        arr = mdarray([1, 2, MISSING_VALUE, 4, 5])
        np_arr = arr.to_numpy()
        self.assertTrue(np.isnan(np_arr[2]))

    def test_elementwise_addition(self):
        """Test element-wise addition with missing values."""
        arr1 = mdarray([1, 2, MISSING_VALUE, 4, 5])
        arr2 = mdarray([5, MISSING_VALUE, 3, 2, 1])
        result = arr1.add(arr2)
        expected = np.array([6, np.nan, np.nan, 6, 6])

        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=5)

    def test_elementwise_multiplication(self):
        """Test element-wise multiplication with missing values."""
        arr1 = mdarray([1, 2, MISSING_VALUE, 4, 5])
        arr2 = mdarray([5, MISSING_VALUE, 3, 2, 1])
        result = arr1.multiply(arr2)
        expected = np.array([5, np.nan, np.nan, 8, 5])

        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=5)

    def test_elementwise_subtraction(self):
        """Test element-wise subtraction with missing values."""
        arr1 = mdarray([10, 20, MISSING_VALUE, 40, 50])
        arr2 = mdarray([5, 10, 15, MISSING_VALUE, 25])
        result = arr1 - arr2
        expected = np.array([5, 10, np.nan, np.nan, 25])

        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=5)

    def test_elementwise_division(self):
        """Test element-wise division, ensuring no division by zero and missing values handled properly."""
        arr1 = mdarray([10, 20, MISSING_VALUE, 40, 50])
        arr2 = mdarray([5, 0, 10, 2, MISSING_VALUE])
        result = arr1 / arr2
        expected = np.array([2, np.nan, np.nan, 20, np.nan])

        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=5)

    def test_log_function(self):
        """Test logarithm function application while preserving missing values."""
        arr = mdarray([1, np.e, MISSING_VALUE, 10])
        result = arr.log()
        expected = np.array([0, 1, np.nan, np.log(10)])

        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=5)

    def test_exp_function(self):
        """Test exponential function application while preserving missing values."""
        arr = mdarray([0, 1, MISSING_VALUE, 2])
        result = arr.exp()
        expected = np.array([1, np.e, np.nan, np.exp(2)])

        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=5)

    def test_trigonometric_functions(self):
        """Test sine and cosine functions with missing values."""
        arr = mdarray([0, np.pi / 2, MISSING_VALUE, np.pi])
        sin_result = arr.sin()
        cos_result = arr.cos()

        expected_sin = np.array([0, 1, np.nan, 0])
        expected_cos = np.array([1, 0, np.nan, -1])

        np.testing.assert_array_almost_equal(sin_result.to_numpy(), expected_sin, decimal=5)
        np.testing.assert_array_almost_equal(cos_result.to_numpy(), expected_cos, decimal=5)


if __name__ == '__main__':
    unittest.main()