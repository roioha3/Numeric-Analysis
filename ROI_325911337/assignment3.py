"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import heapq


class Assignment3:
    def __init__(self):
        """
        Initialization for potential future pre-calculations.
        """
        pass

    # ------------------------------------------------------------------
    # 1. Gauss–Legendre Quadrature
    # ------------------------------------------------------------------
    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Compute the definite integral of f over [a, b]
        using n-point Gauss–Legendre quadrature.
        """

        a = np.float64(a)
        b = np.float64(b)

        # Get nodes and weights in [-1,1]
        nodes, weights = np.polynomial.legendre.leggauss(n)

        # Affine transform to [a,b]
        scale = 0.5 * (b - a)
        shift = 0.5 * (b + a)
        mapped_nodes = scale * nodes + shift

        # Evaluate function
        y_vals = np.array([f(x) for x in mapped_nodes], dtype=np.float64)

        # Weighted sum
        result = scale * np.dot(weights, y_vals)

        return np.float32(result)

    # ------------------------------------------------------------------
    # 2. Area Between Two Functions
    # ------------------------------------------------------------------
    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Compute the total absolute area between f1 and f2.
        The method:
        1. Detect sign changes of f1 - f2
        2. Refine roots via bisection
        3. Integrate absolute area between consecutive intersections
        """

        def diff(x):
            return f1(x) - f2(x)

        # Search interval (kept same logic as original)
        search_range = np.linspace(1.0, 100.0, 1000, dtype=np.float64)
        values = np.array([diff(x) for x in search_range], dtype=np.float64)

        # Detect sign changes
        sign_changes = np.where(np.diff(np.sign(values)) != 0)[0]

        intersections = []

        for idx in sign_changes:
            low = search_range[idx]
            high = search_range[idx + 1]
            f_low = values[idx]

            # Bisection refinement
            for _ in range(25):
                mid = 0.5 * (low + high)
                f_mid = diff(mid)

                if f_low * f_mid <= 0:
                    high = mid
                else:
                    low = mid
                    f_low = f_mid

            root = 0.5 * (low + high)

            # Avoid duplicates
            if not intersections or abs(root - intersections[-1]) > 1e-3:
                intersections.append(root)

        if len(intersections) < 2:
            return np.float32(np.nan)

        # Integrate segment-wise
        total_area = 0.0

        for left, right in zip(intersections[:-1], intersections[1:]):
            segment = self.integrate(diff, left, right, 64)
            total_area += abs(segment)

        return np.float32(total_area)

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()