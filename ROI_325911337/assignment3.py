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
import time
import random

class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        a = np.float32(a)
        b = np.float32(b)

        if n <= 0 or a == b:
            return np.float32(0.0)

        # keep direction consistent
        sign = np.float32(1.0)
        if b < a:
            a, b = b, a
            sign = np.float32(-1.0)

        # Cache to guarantee we never call f more than n times
        cache = {}
        calls = 0

        def _key(x32: np.float32) -> int:
            return int(np.float32(x32).view(np.int32))

        def eval_f(x32: np.float32) -> np.float32:
            nonlocal calls
            x32 = np.float32(x32)
            k = _key(x32)
            if k in cache:
                return cache[k]
            if calls >= n:
                # budget exhausted, should not happen if refinement checks are correct
                return np.float32(0.0)
            y32 = np.float32(f(float(x32)))
            cache[k] = y32
            calls += 1
            return y32

        def simpson_uniform(xs: np.ndarray, ys: np.ndarray) -> np.float32:
            """
            Composite Simpson for uniform grid xs (float32) with an even number of subintervals.
            """
            N = xs.size - 1  # number of subintervals
            h = np.float32((xs[-1] - xs[0]) / np.float32(N))
            s = np.float32(ys[0] + ys[-1])
            if ys.size > 2:
                s += np.float32(4.0) * np.sum(ys[1:-1:2], dtype=np.float32)
                s += np.float32(2.0) * np.sum(ys[2:-1:2], dtype=np.float32)
            return np.float32(h * s / np.float32(3.0))

        # Start with an even number of subintervals (Simpson requirement)
        N = 4
        xs = np.linspace(a, b, N + 1, dtype=np.float32)
        ys = np.array([eval_f(x) for x in xs], dtype=np.float32)

        I_prev = simpson_uniform(xs, ys)

        # Adaptive refinement: each step doubles intervals by inserting midpoints.
        # Adds exactly N new evaluations (the midpoints), so we can check budget safely.
        for _ in range(25):
            if calls + N > n:
                break

            xs_old = xs
            ys_old = ys

            mids = np.float32((xs_old[:-1] + xs_old[1:]) / np.float32(2.0))
            ys_mid = np.array([eval_f(x) for x in mids], dtype=np.float32)

            # Build refined arrays of length (2N + 1) without broadcasting issues
            xs = np.empty((2 * N + 1,), dtype=np.float32)
            ys = np.empty((2 * N + 1,), dtype=np.float32)

            xs[0::2] = xs_old
            ys[0::2] = ys_old
            xs[1::2] = mids
            ys[1::2] = ys_mid

            N *= 2
            I_new = simpson_uniform(xs, ys)

            # Stop when the Simpson refinement is stable enough (float32 scale-aware)
            diff = np.float32(abs(I_new - I_prev))
            scale = np.float32(max(np.float32(1.0), abs(I_new)))
            if diff <= np.float32(1e-5) * scale:
                I_prev = I_new
                break

            I_prev = I_new

        return np.float32(sign * I_prev)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        
        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis
        """

        # Intersections outside [1, 100] may be ignored (per project instructions).
        L = np.float32(1.0)
        R = np.float32(100.0)

        def g(x32: np.float32) -> np.float32:
            return np.float32(f1(float(x32)) - f2(float(x32)))

        # Dense scan to find sign changes
        xs = np.linspace(L, R, 4096, dtype=np.float32)
        gs = np.array([g(x) for x in xs], dtype=np.float32)

        roots = []

        # Bisection on each sign change interval
        for i in range(xs.size - 1):
            a = xs[i]
            b = xs[i + 1]
            fa = gs[i]
            fb = gs[i + 1]

            if fa == 0.0:
                roots.append(a)
                continue

            if fa * fb < 0.0:
                left = a
                right = b
                fL = fa
                for _ in range(60):
                    mid = np.float32((left + right) / np.float32(2.0))
                    fM = g(mid)
                    if abs(fM) <= np.float32(1e-5):
                        left = right = mid
                        break
                    if fL * fM <= 0.0:
                        right = mid
                    else:
                        left = mid
                        fL = fM
                roots.append(np.float32((left + right) / np.float32(2.0)))

        roots = np.unique(np.array(roots, dtype=np.float32))
        roots.sort()

        if roots.size < 2:
            return np.float32(np.nan)

        def absdiff(x: float) -> np.float32:
            x32 = np.float32(x)
            return np.float32(abs(np.float32(f1(float(x32)) - f2(float(x32)))))

        area = np.float32(0.0)
        for i in range(roots.size - 1):
            a = roots[i]
            b = roots[i + 1]
            if b > a:
                # 200 points per segment is usually plenty; you can raise this if needed.
                area = np.float32(area + self.integrate(absdiff, float(a), float(b), 200))

        return np.float32(area)


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
