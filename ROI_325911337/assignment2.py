"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. 
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
        # replace this line with your solution
        def g(x):
            return f1(x) - f2(x)
        
        def refine_root(left, right):
            g_left = g(left)
            g_right = g(right)

            # if one of the ends is already good enough, return it
            if abs(g_left) <= maxerr:
                return left
            if abs(g_right) <= maxerr:
                return right

            # if no sign change and neither endpoint is close to zero, give up on this interval
            if g_left * g_right > 0:
                return None

            for _ in range(50):  # limit iterations
                mid = 0.5 * (left + right)
                g_mid = g(mid)

                if abs(g_mid) <= maxerr:
                    return mid

                # keep the half where the sign changes
                if g_left * g_mid <= 0:
                    right = mid
                    g_right = g_mid
                else:
                    left = mid
                    g_left = g_mid

            # after max iterations, just return the midpoint
            return 0.5 * (left + right)

        N = 1000 # number of sample points
        
        xs = np.linspace(a, b, N)
        ys = [g(x) for x in xs]
        
        intervals = []
        for i in range(N - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            if y0 * y1 <= 0:
                intervals.append((x0, x1))
        
        X = []
        for left, right in intervals:
            root = refine_root(left, right)
            if root is not None:
                if not any(abs(root - r) < 1e-4 for r in X): # To avoid duplicates
                    X.append(root)

        return X


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = lambda x: x - 1 
        f2 = lambda x: 0

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        print(X)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
