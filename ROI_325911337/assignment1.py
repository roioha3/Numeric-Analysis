"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random
import heapq


import numpy as np
from scipy.interpolate import pchip_interpolate

class Assignment1:
    def __init__(self):
        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Fast + accurate black-box interpolation under a strict call budget (<= n).

        Strategy (research-backed):
        - Use piecewise Chebyshev–Lobatto sampling (stable vs. uniform/Runge),
        - Evaluate with barycentric interpolation (numerically stable),
        - Make evaluation O(1) “per x” up to a small constant (k ~ 17),
        and preprocessing O(n).

        This keeps accuracy high on many function families, while staying fast for large n.
        """
        import numpy as np

        # ---------- Edge cases ----------
        if n <= 0:
            return lambda x: 0.0

        if a == b:
            y0 = float(f(a))  # 1 call
            return lambda x, y0=y0: y0

        if n == 1:
            xm = 0.5 * (a + b)
            ym = float(f(xm))  # 1 call
            return lambda x, ym=ym: ym

        # ---------- Choose local degree k (constant work per evaluation) ----------
        # k up to ~17 is a good practical sweet spot (accuracy + speed).
        k = int(min(17, n))
        if k < 2:
            # Shouldn't happen given n>=2, but keep safe.
            xm = 0.5 * (a + b)
            ym = float(f(xm))
            return lambda x, ym=ym: ym

        # With S segments, we reuse boundary sample values, so total calls ≈ 1 + S*(k-1) <= n.
        S = max(1, (n - 1) // (k - 1))
        seg_len = (b - a) / S

        # Reference Chebyshev–Lobatto nodes on [-1,1]
        j = np.arange(k, dtype=np.float64)
        t = np.cos(np.pi * j / (k - 1))  # descending from 1 to -1

        # Barycentric weights for Chebyshev–Lobatto (up to common scaling)
        w = np.ones(k, dtype=np.float64)
        w[0] = 0.5
        w[-1] = 0.5
        w *= (-1.0) ** j  # alternating signs

        # ---------- Precompute per-segment nodes and samples (<= n calls total) ----------
        xs_list = []
        ys_list = []

        # We store per-segment arrays for fast evaluation
        seg_xs = []
        seg_ys = []

        calls = 0
        prev_right_y = None

        for s in range(S):
            left = a + s * seg_len
            right = a + (s + 1) * seg_len

            # Map nodes from [-1,1] to [left,right]
            xs = 0.5 * (left + right) + 0.5 * (right - left) * t

            ys = np.empty(k, dtype=np.float64)

            # Reuse boundary point: left endpoint of this segment equals right endpoint of previous segment.
            # Cheb–Lobatto includes endpoints: xs[0]=right, xs[-1]=left (because t descends 1..-1).
            # The shared boundary is at 'left' == previous 'right', which is xs[-1] here.
            # For segment s>0, xs_prev[0] was previous right endpoint, which equals current left endpoint.
            # Since our nodes are descending, current left endpoint is xs[-1].
            if s == 0:
                # Sample all k points
                for idx in range(k):
                    if calls >= n:
                        break
                    yv = float(f(float(xs[idx])))
                    ys[idx] = yv
                    calls += 1
                # If we broke early (shouldn't with our S choice), fill remaining by nearest already-sampled value
                if calls < (s + 1) * k and calls < n:
                    # not expected; keep safe
                    ys[np.isnan(ys)] = ys[0]
            else:
                # Reuse current left endpoint sample (xs[-1]) from previous segment's right endpoint (prev xs[0])
                ys[-1] = float(prev_right_y)

                # Sample remaining k-1 points (exclude xs[-1])
                for idx in range(k - 1):
                    if calls >= n:
                        # If budget is unexpectedly tight, mirror from closest sampled point
                        ys[idx] = ys[-1]
                        continue
                    yv = float(f(float(xs[idx])))
                    ys[idx] = yv
                    calls += 1

            # Remember this segment's right endpoint value for reuse in next segment.
            # Right endpoint in descending node order is xs[0].
            prev_right_y = ys[0]

            seg_xs.append(xs)
            seg_ys.append(ys)

        seg_xs = np.array(seg_xs, dtype=np.float64)  # shape (S, k)
        seg_ys = np.array(seg_ys, dtype=np.float64)  # shape (S, k)

        # ---------- Evaluation (O(k) per x, k is small constant) ----------
        # Small epsilon to detect "x equals a node" robustly
        eps = 1e-14

        def bary_eval(xv: float, si: int) -> float:
            xs = seg_xs[si]
            ys = seg_ys[si]
            diff = xv - xs

            # If xv hits a node, return exact sample
            hit = np.where(np.abs(diff) <= eps)[0]
            if hit.size > 0:
                return float(ys[int(hit[0])])

            inv = w / diff
            num = np.sum(inv * ys)
            den = np.sum(inv)
            return float(num / den)

        def result(x):
            # Scalar fast-path
            if np.ndim(x) == 0:
                xv = float(x)
                if xv <= a:
                    return bary_eval(a, 0)
                if xv >= b:
                    return bary_eval(b, S - 1)
                si = int((xv - a) / seg_len)
                if si < 0:
                    si = 0
                elif si >= S:
                    si = S - 1
                return bary_eval(xv, si)

            # Vector path
            xx = np.asarray(x, dtype=np.float64)
            out = np.empty_like(xx, dtype=np.float64)

            # Clamp
            xx_clamped = np.clip(xx, a, b)

            seg_idx = ((xx_clamped - a) / seg_len).astype(np.int64)
            np.clip(seg_idx, 0, S - 1, out=seg_idx)

            # Evaluate per segment in batches
            for si in range(S):
                mask = (seg_idx == si)
                if not np.any(mask):
                    continue
                xvals = xx_clamped[mask]

                xs = seg_xs[si]
                ys = seg_ys[si]

                # Compute barycentric for each x in this segment
                # (loop over points; k is small, and points per segment are typically moderate)
                vals = np.empty_like(xvals, dtype=np.float64)
                for i_pt, xv in enumerate(xvals):
                    diff = xv - xs
                    hit = np.where(np.abs(diff) <= eps)[0]
                    if hit.size > 0:
                        vals[i_pt] = ys[int(hit[0])]
                    else:
                        inv = w / diff
                        vals[i_pt] = np.sum(inv * ys) / np.sum(inv)

                out[mask] = vals

            return out

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
