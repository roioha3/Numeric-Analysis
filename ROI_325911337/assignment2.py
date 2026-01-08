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
        if a == b:
            x = float(a)
            return [x] if abs(f1(x) - f2(x)) <= maxerr else []

        if a > b:
            a, b = b, a

        def g(x):
            return f1(x) - f2(x)

        base = 2048
        span = b - a
        N = int(np.clip(base * np.sqrt(max(1.0, span)), 512, 20000))

        xs = np.linspace(a, b, N, dtype=float)

        try:
            y = g(xs)
            y = np.asarray(y, dtype=float)
            if y.shape != xs.shape:
                raise ValueError
        except Exception:
            y = np.fromiter((g(x) for x in xs), dtype=float, count=N)

        eps = float(maxerr)
        mask_close = np.abs(y) <= eps
        idx_close = np.flatnonzero(mask_close)

        idx_sc = np.flatnonzero((y[:-1] * y[1:]) <= 0.0)

        candidates = np.unique(np.concatenate((idx_sc, idx_close)))
        if candidates.size == 0:
            return []

        def refine(left, right, gl, gr):
            if abs(gl) <= eps:
                return left
            if abs(gr) <= eps:
                return right
            if gl == 0.0:
                return left
            if gr == 0.0:
                return right
            if gl * gr > 0.0:
                return None

            xl, xr = left, right
            fl, fr = gl, gr
            x = 0.5 * (xl + xr)
            fx = g(x)

            for _ in range(30):
                if abs(fx) <= eps:
                    return x

                denom = (fr - fl)
                if denom != 0.0:
                    xs_ = xr - fr * (xr - xl) / denom
                else:
                    xs_ = 0.5 * (xl + xr)

                if not (xl < xs_ < xr):
                    xs_ = 0.5 * (xl + xr)

                fs = g(xs_)
                if abs(fs) <= eps:
                    return xs_

                if fl * fs <= 0.0:
                    xr, fr = xs_, fs
                else:
                    xl, fl = xs_, fs

                x = 0.5 * (xl + xr)
                fx = g(x)

            return x

        roots = []

        for i in idx_close:
            roots.append(float(xs[i]))

        for i in idx_sc:
            left = float(xs[i])
            right = float(xs[i + 1])
            gl = float(y[i])
            gr = float(y[i + 1])
            r = refine(left, right, gl, gr)
            if r is not None:
                roots.append(float(r))

        if not roots:
            return []

        roots.sort()
        out = [roots[0]]
        tol = max(1e-6, 10.0 * eps, (b - a) / max(1.0, N) * 0.25)
        for x in roots[1:]:
            if abs(x - out[-1]) > tol:
                out.append(x)

        final = []
        for x in out:
            if abs(f1(x) - f2(x)) <= eps:
                final.append(x)

        return final

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

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
