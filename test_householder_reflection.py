import unittest
from numpy.linalg import matrix_rank
import numpy as np
import numpy.testing as npt

from math import copysign, hypot



def householder_reflection(A):
    """Perform QR decomposition of matrix A using Householder reflection."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    # Iterative over column sub-vector and
    # compute Householder matrix to zero-out lower triangular matrix entries.
    for cnt in range(num_rows - 1):
        x = R[cnt:, cnt]

        e = np.zeros_like(x)
        e[0] = copysign(np.linalg.norm(x), -A[cnt, cnt])
        u = x + e
        v = u / np.linalg.norm(u)

        Q_cnt = np.identity(num_rows)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)

        R = np.dot(Q_cnt, R)
        Q = np.dot(Q, Q_cnt.T)
    
    return (Q, R)

class TestHouseholderReflection(unittest.TestCase):
    """Test case for QR decomposition using Householder reflection."""

    def test_wikipedia_example1(self):
        """Test of Wikipedia example
        The example for the following QR decomposition is taken from
        https://en.wikipedia.org/wiki/Qr_decomposition#Example_2.
        """

        A = np.array([[12, -51, 4],
                      [6, 167, -68],
                      [-4, 24, -41]], dtype=np.float64)
        print(" Matrix rank",matrix_rank(A))
        (Q, R) = householder_reflection(A)

        Q_desired = np.array([[0.8571, -0.3943, 0.3314],
                              [0.4286, 0.9029, -0.0343],
                              [-0.2857, 0.1714, 0.9429]], dtype=np.float64)
        R_desired = np.array([[14, 21, -14],
                              [0, 175, -70],
                              [0, 0, -35]], dtype=np.float64)
        print(R)
        npt.assert_almost_equal(Q, Q_desired, 4)
        npt.assert_almost_equal(R, R_desired, 4)


if __name__ == "__main__":
    unittest.main()
