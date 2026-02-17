"""
In this assignment you should interpolate the given function.
"""

import numpy as np


class Assignment1:
    def __init__(self):
        """
        Place for one-time precomputations if needed.
        """
        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate f on [a,b] using at most n points.
        Uses Chebyshev nodes (first kind) and barycentric interpolation.
        """

        n = int(n)
        if n <= 0:
            return lambda x: np.zeros_like(x, dtype=float) if np.ndim(x) > 0 else 0.0

        # ------------------------------------------------------------------
        # 1. Chebyshev nodes (first kind)
        # ------------------------------------------------------------------
        k = np.arange(n)
        theta = (2 * k + 1) * np.pi / (2 * n)
        x_cheb = np.cos(theta)

        # Map to [a,b]
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        x_nodes = mid + half * x_cheb

        # ------------------------------------------------------------------
        # 2. Sample function (ONLY n calls)
        # ------------------------------------------------------------------
        y_nodes = np.array([f(x) for x in x_nodes], dtype=np.float64)

        # ------------------------------------------------------------------
        # 3. Barycentric weights for Chebyshev (first kind)
        # Simplified stable form
        # ------------------------------------------------------------------
        weights = (-1.0) ** k

        # ------------------------------------------------------------------
        # 4. Interpolator
        # ------------------------------------------------------------------
        def g(x_eval):
            x = np.asarray(x_eval, dtype=np.float64)
            scalar = x.ndim == 0
            if scalar:
                x = x[None]

            # Differences matrix
            diffs = x[:, None] - x_nodes[None, :]

            # Detect exact node hits (important for stability)
            mask = np.isclose(diffs, 0.0, atol=1e-14)

            # Avoid division by zero
            safe_diffs = np.where(mask, 1.0, diffs)

            temp = weights / safe_diffs
            numerator = temp @ y_nodes
            denominator = np.sum(temp, axis=1)

            result = numerator / denominator

            # Patch exact matches
            if np.any(mask):
                rows, cols = np.where(mask)
                result[rows] = y_nodes[cols]

            return result[0] if scalar else result

        return g


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)



if __name__ == "__main__":
    unittest.main()
