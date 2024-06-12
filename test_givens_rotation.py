import unittest

import numpy as np
import numpy.testing as npt

from math import copysign, hypot

def givens_rotation(A):
    """Perform QR decomposition of matrix A using Givens rotation."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    for (row, col) in zip(rows, cols):

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        if R[row, col] != 0:
            (c, s) = _givens_rotation_matrix_entries(R[col, col], R[row, col])

            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)
            

    return (Q, R)

def _givens_rotation_matrix_entries(a, b):
    """Compute matrix entries for Givens rotation."""
    r = hypot(a, b)
    c = a/r
    s = -b/r

    return (c, s)

class TestGivensRotation(unittest.TestCase):
    """Test case for QR decomposition using Givens rotation."""

    def test_wikipedia_example1(self):
        """Test of Wikipedia example
        The example for the following QR decomposition is taken from
        https://en.wikipedia.org/wiki/Givens_rotation#Triangularization.
        """

        A = np.array([[6, 5, 0],
                      [5, 1, 4],
                      [0, 4, 3]], dtype=np.float64)

        (Q, R) = givens_rotation(A)

        Q_desired = np.array([[0.7682, 0.3327, 0.5470],
                              [0.6402, -0.3992, -0.6564],
                              [0, 0.8544, -0.5196]], dtype=np.float64)
        R_desired = np.array([[7.8102, 4.4813, 2.5607],
                              [0, 4.6817, 0.9664],
                              [0, 0, -4.1843]], dtype=np.float64)

        npt.assert_almost_equal(Q, Q_desired, 4)
        npt.assert_almost_equal(R, R_desired, 4)

    def test_wikipedia_example2(self):
        """Test of Wikipedia example
        The example for the following QR decomposition is taken from
        http://de.wikipedia.org/wiki/Givens-Rotation.
        """

        A = np.array([[3, 5],
                      [0, 2],
                      [0, 0],
                      [4, 5]], dtype=np.float64)

        (Q, R) = givens_rotation(A)

        Q_desired = np.array([[0.6, 0.3577, 0, -0.7155],
                              [0, 0.8944, 0, 0.4472],
                              [0, 0, 1, 0],
                              [0.8, -0.2683, 0, 0.5366]], dtype=np.float64)
        R_desired = np.array([[5, 7],
                              [0, 2.2360],
                              [0, 0],
                              [0, 0]], dtype=np.float64)

        npt.assert_almost_equal(Q, Q_desired, 4)
        npt.assert_almost_equal(R, R_desired, 4)


if __name__ == "__main__":
    unittest.main()