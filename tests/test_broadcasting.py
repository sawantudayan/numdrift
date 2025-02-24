import unittest
import numpy as np
from numdrift.core import mdarray, MISSING_VALUE

class TestBroadcasting(unittest.TestCase):
    """
    Unit tests for broadcasting support in mdarray.
    """

    def test_scalar_broadcasting_add(self):
        """Test broadcasting a scalar to an array for addition."""
        arr = mdarray([1, 2, 3])
        result = arr.add(5)
        expected = np.array([6, 7, 8])

        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_scalar_broadcasting_multiply(self):
        """Test broadcasting a scalar to an array for multiplication."""
        arr = mdarray([1, 2, 3])
        result = arr.multiply(2)
        expected = np.array([2, 4, 6])

        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_array_broadcasting_add(self):
        """Test broadcasting between arrays of different shapes for addition."""
        arr1 = mdarray([[1, 2, 3], [4, 5, 6]])
        arr2 = mdarray([10, 20, 30])  # Will be broadcasted

        result = arr1.add(arr2)
        expected = np.array([[11, 22, 33], [14, 25, 36]])

        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_array_broadcasting_multiply(self):
        """Test broadcasting between arrays for multiplication."""
        arr1 = mdarray([[1, 2, 3], [4, 5, 6]])
        arr2 = mdarray([[2], [3]])  # Broadcast along columns

        result = arr1.multiply(arr2)
        expected = np.array([[2, 4, 6], [12, 15, 18]])

        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_broadcasting_with_missing_values(self):
        """Test broadcasting when arrays contain missing values."""
        arr1 = mdarray([1, MISSING_VALUE, 3])
        arr2 = mdarray([[10], [20], [30]])  # Broadcast along rows

        result = arr1.add(arr2)
        expected = np.array([[11, np.nan, 13],
                             [21, np.nan, 23],
                             [31, np.nan, 33]])

        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_incompatible_shapes(self):
        """Test broadcasting failure on incompatible shapes."""
        arr1 = mdarray([[1, 2], [3, 4]])
        arr2 = mdarray([1, 2, 3])  # Incompatible shape

        with self.assertRaises(ValueError):
            _ = arr1.add(arr2)

    def test_broadcasting_division(self):
        """Test broadcasting with division and missing values."""
        arr1 = mdarray([[10, 20], [30, 40]])
        arr2 = mdarray([2, MISSING_VALUE])  # Broadcast across rows

        result = arr1.divide(arr2)
        expected = np.array([[5, np.nan], [15, np.nan]])

        np.testing.assert_array_equal(result.to_numpy(), expected)

if __name__ == '__main__':
    unittest.main()
