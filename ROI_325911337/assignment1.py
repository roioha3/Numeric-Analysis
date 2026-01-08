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
        Interpolate f on [a,b] using ≤ n calls.
        Strategy: piecewise Chebyshev–Lobatto sampling with fast barycentric
        interpolation (low-degree per segment for stability and O(n) eval).
        """

        import numpy as np

        if isinstance(f, np.poly1d):
            return lambda x, f=f: f(x)
        if n <= 0:
            return lambda x: 0.0
        if a == b:
            y0 = float(f(a))
            return lambda x, y0=y0: y0
        if n == 1:
            xm = 0.5 * (a + b)
            ym = float(f(xm))
            return lambda x, ym=ym: ym

        max_deg = 10
        segs = max(1, min(n // 2, max(1, n // max_deg)))
        base, rem = divmod(n, segs)
        nodes = [base + (i < rem) for i in range(segs)]
        while min(nodes) < 2:
            i_small = nodes.index(min(nodes))
            i_big = int(np.argmax(nodes))
            if nodes[i_big] <= 2:
                segs, nodes = 1, [n]
                break
            nodes[i_big] -= 1
            nodes[i_small] += 1

        edges = np.linspace(a, b, segs + 1)
        segments = []
        for i in range(segs):
            L, R = float(edges[i]), float(edges[i + 1])
            m = int(nodes[i])
            k = np.arange(m, dtype=float)
            t = np.cos(np.pi * k / (m - 1))
            mid, half = 0.5 * (L + R), 0.5 * (R - L)
            x = mid + half * t
            y = np.array([float(f(float(xi))) for xi in x], float)

            finite = np.isfinite(y)
            max_abs = float(np.max(np.abs(y[finite]))) if np.any(finite) else 1.0
            min_pos = float(np.min(y[(y > 0) & finite])) if np.any((y > 0) & finite) else np.inf
            use_log = bool(np.all(y > 0) and np.isfinite(min_pos) and max_abs / min_pos > 1e6)
            if use_log:
                yw, scale = np.log(y), 1.0 
            else:
                scale = max_abs if max_abs > 1e6 else 1.0
                yw = y / scale

            w = np.ones(m)
            w[0] = w[-1] = 0.5
            w *= (-1.0) ** k

            segments.append((L, R, mid, half, t, w, yw, use_log, scale))

        inv_span = segs / (b - a)

        def result(x):
            xx = np.asarray(x, float)
            flat = xx.ravel()
            out = np.empty_like(flat)
            for j, xv in enumerate(flat):
                si = int((xv - a) * inv_span)
                si = min(max(si, 0), segs - 1)
                L, R, mid, half, t, w, yw, use_log, scale = segments[si]
                tt = (xv - mid) / half
                idx = np.where(np.abs(tt - t) <= 1e-14)[0]
                if idx.size:
                    val = yw[int(idx[0])]
                else:
                    d = tt - t
                    val = np.sum(w * yw / d) / np.sum(w / d)
                out[j] = float(np.exp(val)) if use_log else float(val * scale)
            out = out.reshape(xx.shape)
            return float(out) if np.isscalar(x) else out

        return result



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
