import unittest
import numpy as np
from numdrift.core import mdarray, MISSING_VALUE

class TestAdvancedIndexing(unittest.TestCase):
    """
    Unit tests for advanced indexing in mdarray.
    Covers slicing, fancy indexing, and boolean indexing.
    """

    def setUp(self):
        """Initialize an mdarray for testing."""
        self.arr = mdarray([10, 20, MISSING_VALUE, 40, 50])

    def test_basic_slicing(self):
        """Test simple slicing behavior."""
        sliced = self.arr[1:4]
        expected_data = np.array([20, np.nan, 40])
        expected_mask = np.array([False, True, False])

        np.testing.assert_array_equal(sliced.data, expected_data)
        np.testing.assert_array_equal(sliced.mask, expected_mask)

    def test_fancy_indexing(self):
        """Test fancy indexing using lists/arrays."""
        indexed = self.arr[[0, 2, 4]]
        expected_data = np.array([10, np.nan, 50])
        expected_mask = np.array([False, True, False])

        np.testing.assert_array_equal(indexed.data, expected_data)
        np.testing.assert_array_equal(indexed.mask, expected_mask)

    def test_boolean_indexing(self):
        """Test boolean indexing to filter elements."""
        mask = np.array([True, False, False, True, False])
        filtered = self.arr[mask]
        expected_data = np.array([10, 40])
        expected_mask = np.array([False, False])

        np.testing.assert_array_equal(filtered.data, expected_data)
        np.testing.assert_array_equal(filtered.mask, expected_mask)

    def test_setitem_slicing(self):
        """Test setting values using slicing."""
        self.arr[1:3] = [100, MISSING_VALUE]
        expected_data = np.array([10, 100, np.nan, 40, 50])
        expected_mask = np.array([False, False, True, False, False])

        np.testing.assert_array_equal(self.arr.data, expected_data)
        np.testing.assert_array_equal(self.arr.mask, expected_mask)

    def test_setitem_fancy_indexing(self):
        """Test setting values using fancy indexing."""
        self.arr[[0, 4]] = [200, 300]
        expected_data = np.array([200, 20, np.nan, 40, 300])
        expected_mask = np.array([False, False, True, False, False])

        np.testing.assert_array_equal(self.arr.data, expected_data)
        np.testing.assert_array_equal(self.arr.mask, expected_mask)

    def test_setitem_boolean_indexing(self):
        """Test setting values using boolean indexing."""
        mask = np.array([True, False, False, True, False])
        self.arr[mask] = MISSING_VALUE
        expected_data = np.array([np.nan, 20, np.nan, np.nan, 50])
        expected_mask = np.array([True, False, True, True, False])

        np.testing.assert_array_equal(self.arr.data, expected_data)
        np.testing.assert_array_equal(self.arr.mask, expected_mask)


if __name__ == '__main__':
    unittest.main()
