"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random
import heapq


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Adaptive piecewise-linear interpolation (no detectors), using <= n calls to f.
        """
        import numpy as np
        import heapq

        if n <= 0:
            return lambda x: 0.0
        if a == b:
            y0 = float(f(a))
            return lambda x, y0=y0: y0

        samples = {}  # x -> f(x)

        def sample(x):
            x = float(x)
            if x in samples:
                return samples[x]
            if len(samples) >= n:
                return None
            y = float(f(x))
            samples[x] = y
            return y

        # endpoints
        ya = sample(a)
        if ya is None:
            return lambda x: 0.0
        if n == 1:
            return lambda x, ya=ya: ya
        yb = sample(b)
        if yb is None:
            return lambda x, ya=ya: ya

        heap = []

        def push_interval(x0, x1):
            xm = 0.5 * (x0 + x1)
            ym = sample(xm)
            if ym is None:
                return
            y0, y1 = samples[x0], samples[x1]
            ylin = y0 + (y1 - y0) * (xm - x0) / (x1 - x0)
            pr = abs(ym - ylin) * (x1 - x0)
            heapq.heappush(heap, (-pr, x0, x1))

        xa, xb = float(a), float(b)
        push_interval(xa, xb)

        while heap and len(samples) < n:
            _, x0, x1 = heapq.heappop(heap)
            xm = 0.5 * (x0 + x1)
            if xm not in samples:
                if sample(xm) is None:
                    break
            push_interval(x0, xm)
            push_interval(xm, x1)

        xs = np.array(sorted(samples.keys()), dtype=float)
        ys = np.array([samples[x] for x in xs], dtype=float)

        def interp(z, xs=xs, ys=ys):
            z = float(z)
            if z <= xs[0]:
                return float(ys[0])
            if z >= xs[-1]:
                return float(ys[-1])
            i = int(np.searchsorted(xs, z) - 1)
            if i < 0:
                i = 0
            if i >= len(xs) - 1:
                i = len(xs) - 2
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            return float(y0 + (y1 - y0) * (z - x0) / (x1 - x0))

        return interp



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


# Run ONLY the failing case: a=1, b=3, n=10, function=f8
# Put this in the same file / environment where:
#   1) your Assignment1 class exists (with interpolate implemented)
#   2) f8 is defined
#
# It prints the average absolute error on 2*n random points (as described).

import random
import numpy as np
from math import e, log

def run_single_interpolation_test(Assignment1Class, f8, a=1.0, b=3.0, n=10, seed=0):
    rng = random.Random(seed)

    ass = Assignment1Class()
    g = ass.interpolate(f8, a, b, n)  # interpolating function

    # 2*n random test points in [a,b]
    xs = [a + (b - a) * rng.random() for _ in range(2 * n)]

    # average absolute error
    errs = []
    for x in xs:
        fx = float(f8(x))
        gx = float(g(x))
        errs.append(abs(fx - gx))

    avg_err = sum(errs) / len(errs)
    max_err = max(errs)

    print(f"a={a}, b={b}, n={n}, seed={seed}")
    print(f"avg_abs_error = {avg_err}")
    print(f"max_abs_error = {max_err}")

    # optional: show worst point
    worst_i = max(range(len(xs)), key=lambda i: errs[i])
    print(f"worst_x = {xs[worst_i]}")
    print(f"f8(worst_x) = {float(f8(xs[worst_i]))}")
    print(f"g(worst_x)  = {float(g(xs[worst_i]))}")

    return avg_err, max_err

# Example usage:
# run_single_interpolation_test(Assignment1, f8, a=1, b=3, n=10, seed=0)
# run_single_interpolation_test(Assignment1, f8, a=1, b=3, n=10, seed=1)

if __name__ == "__main__":
    #unittest.main()
    def f8(x):
        return pow(e, pow(e, x))   # exp(exp(x))
    run_single_interpolation_test(Assignment1, f8, a=1, b=3, n=10, seed=0)
    run_single_interpolation_test(Assignment1, f8, a=1, b=3, n=10, seed=1)
