import unittest
import numpy as np
from numdrift.core import mdarray, MISSING_VALUE


class TestMDArray(unittest.TestCase):

    def test_array_creation(self):
        arr = mdarray([1, 2, None, 4, 5])
        self.assertEqual(arr.data.tolist(), [1, 2, 0, 4, 5])
        self.assertTrue(arr.mask[2])

    def test_indexing(self):
        arr = mdarray([1, 2, None, 4, 5])
        self.assertEqual(arr[1], 2)
        self.assertIsNone(arr[2])

    def test_slicing(self):
        arr = mdarray([1, 2, None, 4, 5])
        sliced = arr.slice(1, 4)
        self.assertEqual(sliced.data.tolist(), [2, 0, 4])
        self.assertTrue(sliced.mask[1])

    def test_setitem(self):
        arr = mdarray([1, 2, None, 4, 5])
        arr[3] = None
        self.assertTrue(arr.mask[3])

    def test_numpy_conversion(self):
        arr = mdarray([1, 2, None, 4, 5])
        np_arr = arr.to_numpy()
        self.assertTrue(np.isnan(np_arr[2]))


if __name__ == '__main__':
    unittest.main()
