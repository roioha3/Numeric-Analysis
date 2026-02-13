"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment.
"""

import numpy as np
import time


def _poly_eval_inc(coeffs_inc: np.ndarray, t):
    """
    Evaluate polynomial sum_{k=0}^d coeffs_inc[k] * t^k using Horner.
    Works for scalar or numpy array t.
    """
    res = 0.0
    for c in coeffs_inc[::-1]:
        res = res * t + c
    return res


def _cholesky_solve_spd(A: np.ndarray, b: np.ndarray):
    """
    Solve Ax=b for SPD A using a small custom Cholesky.
    Returns None if A is not SPD numerically.
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)

    # A = L L^T
    for i in range(n):
        for j in range(i + 1):
            s = float(np.dot(L[i, :j], L[j, :j]))
            if i == j:
                v = float(A[i, i]) - s
                if v <= 0.0 or not np.isfinite(v):
                    return None
                L[i, j] = np.sqrt(v)
            else:
                diag = float(L[j, j])
                if diag <= 0.0 or not np.isfinite(diag):
                    return None
                L[i, j] = (float(A[i, j]) - s) / diag

    # Forward: L y = b
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        diag = float(L[i, i])
        if diag == 0.0 or not np.isfinite(diag):
            return None
        y[i] = (float(b[i]) - float(np.dot(L[i, :i], y[:i]))) / diag

    # Backward: L^T x = y
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        diag = float(L[i, i])
        if diag == 0.0 or not np.isfinite(diag):
            return None
        x[i] = (float(y[i]) - float(np.dot(L[i + 1:, i], x[i + 1:]))) / diag

    return x


class Assignment4:
    def __init__(self):
        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        start = time.time()

        maxtime = float(maxtime) if maxtime is not None else 1.0
        if maxtime <= 0.0:
            return lambda x: 0.0

        # normalize order
        if b < a:
            a, b = b, a

        # degenerate interval
        if a == b:
            try:
                y0 = float(np.asarray(f(a)).reshape(-1)[0])
            except Exception:
                y0 = 0.0

            def g(x, c=y0):
                if isinstance(x, np.ndarray):
                    return np.full_like(x, c, dtype=np.float64)
                return float(c)

            return g

        deg = int(d) if d is not None else 0
        if deg < 0:
            deg = 0

        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)  # > 0 because a != b

        # EXTRA SAFETY: never divide by 0 (shouldn't happen here, but keep it)
        if half == 0.0 or not np.isfinite(half):
            try:
                c0 = float(np.asarray(f(mid)).reshape(-1)[0])
            except Exception:
                c0 = 0.0
            return (lambda x, c=c0: np.full_like(x, c, dtype=np.float64)) if isinstance(x, np.ndarray) else (lambda x, c=c0: float(c))

        # stop sampling slightly before maxtime to leave time for solve/closure
        reserve = min(0.05 + 0.005 * (deg + 1), maxtime * 0.10)
        reserve = max(0.01, reserve)  # keep a tiny reserve always
        deadline = start + max(0.0, maxtime - reserve)

        # probe point in middle of interval
        x_probe = float(mid)
        t0 = time.time()
        try:
            y_probe = f(x_probe)
        except Exception:
            return lambda x: 0.0
        t1 = time.time()

        y_probe = float(np.asarray(y_probe).reshape(-1)[0])
        eval_dt = max(1e-6, t1 - t0)

        # estimate noise scale cheaply (same point repeated)
        noise_samples = [y_probe]
        max_probe_time = min(0.1 * maxtime, 0.25)
        while (time.time() - start) < max_probe_time and len(noise_samples) < 20:
            if time.time() + eval_dt > deadline:
                break
            t0 = time.time()
            try:
                yy = f(x_probe)
            except Exception:
                break
            t1 = time.time()
            yy = float(np.asarray(yy).reshape(-1)[0])
            noise_samples.append(yy)
            eval_dt = 0.8 * eval_dt + 0.2 * max(1e-6, (t1 - t0))

        sigma = float(np.std(noise_samples)) if len(noise_samples) > 1 else 0.0

        # detect vectorization + per-element noise
        vector_ok = False
        noise_per_element = False
        if time.time() + eval_dt < deadline:
            try:
                arr = np.full(8, x_probe, dtype=np.float64)
                yarr = np.asarray(f(arr))
                if yarr.shape == arr.shape:
                    vector_ok = True
                    # if noise is injected per element, identical x values differ
                    thresh = (1e-10 if sigma == 0 else 0.05 * max(1e-12, sigma))
                    noise_per_element = bool(np.std(yarr) > thresh)
            except Exception:
                vector_ok = False
                noise_per_element = False

        use_vector = bool(vector_ok and noise_per_element)

        # accumulate normal equations for LS on t=(x-mid)/half in [-1,1]
        n_params = deg + 1
        if n_params <= 0:
            return lambda x: 0.0

        A = np.zeros((n_params, n_params), dtype=np.float64)
        bvec = np.zeros(n_params, dtype=np.float64)
        n_used = 0

        def add_scalar_sample(tt: float, yy: float):
            nonlocal n_used, A, bvec
            # v = [1, t, t^2, ...]
            v = np.empty(n_params, dtype=np.float64)
            v[0] = 1.0
            for k in range(1, n_params):
                v[k] = v[k - 1] * tt

            # faster / cleaner: full outer product (still same LS steps)
            bvec += v * yy
            A += np.outer(v, v)
            n_used += 1

        # include probe point
        add_scalar_sample((x_probe - mid) / half, y_probe)

        # cap samples to avoid insane loops when f is extremely fast.
        cap_samples = 50000

        time_left = max(0.0, deadline - time.time())
        max_scalar_calls = int(time_left / eval_dt) if eval_dt > 0 else 0
        max_scalar_calls = max(0, max_scalar_calls)

        target_samples = min(max_scalar_calls, cap_samples)
        target_samples = max(target_samples, min(max_scalar_calls, 10 * n_params))
        remaining = max(0, target_samples - 1)

        rng = np.random.default_rng()

        # vector sampling
        if use_vector and remaining > 0:
            # keep batches moderate to avoid big V matrices
            batch = min(8000, max(1000, 150 * n_params))

            while remaining > 0 and time.time() + eval_dt < deadline:
                m = int(min(batch, remaining))
                if m <= 0:
                    break

                xs = rng.uniform(a, b, size=m).astype(np.float64)

                t0 = time.time()
                try:
                    ys = np.asarray(f(xs), dtype=np.float64)
                except Exception:
                    use_vector = False
                    break
                t1 = time.time()

                if ys.shape != xs.shape:
                    use_vector = False
                    break

                dt = max(1e-6, (t1 - t0))
                eval_dt = 0.8 * eval_dt + 0.2 * dt

                ts = (xs - mid) / half
                # same steps: build design matrix then normal eq
                V = np.vander(ts, N=n_params, increasing=True)  # (m, p)
                A += V.T @ V
                bvec += V.T @ ys
                n_used += m
                remaining -= m

        # scalar sampling fallback
        if (not use_vector) and remaining > 0:
            m = int(min(remaining, cap_samples))
            if m > 0:
                xs = np.linspace(a, b, num=m + 2, dtype=np.float64)[1:-1]
                jitter = (b - a) * 1e-12
                if jitter != 0.0 and np.isfinite(jitter):
                    xs = xs + rng.uniform(-jitter, jitter, size=xs.shape)

                for x in xs:
                    if time.time() + eval_dt > deadline:
                        break

                    t0 = time.time()
                    try:
                        y = f(float(x))
                    except Exception:
                        break
                    t1 = time.time()

                    y = float(np.asarray(y).reshape(-1)[0])
                    if not np.isfinite(y):
                        continue

                    eval_dt = 0.8 * eval_dt + 0.2 * max(1e-6, (t1 - t0))
                    add_scalar_sample((float(x) - mid) / half, y)

        # ensure symmetry (numerical)
        A = 0.5 * (A + A.T)

        # if too few samples, reduce degree
        if n_used <= deg:
            deg = max(0, n_used - 1)
            n_params = deg + 1
            if n_params <= 0:
                return lambda x: 0.0
            A = A[:n_params, :n_params]
            bvec = bvec[:n_params]

        # solve ridge regression for stability
        tr = float(np.trace(A))
        scale = tr / n_params if tr > 0 and np.isfinite(tr) else 1.0
        lam = (1e-12 + 1e-6 * (sigma ** 2)) * scale
        if n_used < 5 * n_params:
            lam = max(lam, 1e-8 * scale)

        coeffs = None
        A_work = A.copy()

        for _ in range(10):
            A_work[:, :] = A
            A_work.flat[::n_params + 1] += lam  # add ridge on diagonal
            coeffs = _cholesky_solve_spd(A_work, bvec)
            if coeffs is not None and np.all(np.isfinite(coeffs)):
                break
            lam *= 10.0

        if coeffs is None:
            c0 = float(np.median(noise_samples)) if noise_samples else y_probe

            def g(x, c=c0):
                if isinstance(x, np.ndarray):
                    return np.full_like(x, c, dtype=np.float64)
                return float(c)

            return g

        coeffs = np.asarray(coeffs, dtype=np.float64)

        # trim tiny high-degree tail (helps when d is over-estimated)
        if coeffs.size > 1:
            cscale = float(np.max(np.abs(coeffs))) if np.any(coeffs) else 0.0
            thr = max(1e-10, 5e-3 * (sigma if sigma > 0 else 1e-3), 1e-8 * cscale)
            new_deg = coeffs.size - 1
            while new_deg > 0 and abs(coeffs[new_deg]) < thr:
                new_deg -= 1
            coeffs = coeffs[:new_deg + 1]

        mid_c, half_c, coeffs_c = float(mid), float(half), coeffs

        # final fitted callable
        def fitted(x):
            # avoid divide-by-0 defensively (should not happen)
            if half_c == 0.0 or not np.isfinite(half_c):
                if isinstance(x, np.ndarray):
                    return np.full_like(x, float(coeffs_c[0]), dtype=np.float64)
                return float(coeffs_c[0])

            x_arr = np.asarray(x, dtype=np.float64)
            t = (x_arr - mid_c) / half_c
            y = _poly_eval_inc(coeffs_c, t)

            if np.isscalar(x):
                return float(np.asarray(y).reshape(-1)[0])
            return y

        return fitted

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEqual(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()