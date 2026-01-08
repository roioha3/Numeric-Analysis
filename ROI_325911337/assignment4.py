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
import random


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    @staticmethod
    def _solve_gauss_pp(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        A = A.astype(np.float64, copy=True)
        b = b.astype(np.float64, copy=True)
        n = A.shape[0]

        for k in range(n):
            piv = k + int(np.argmax(np.abs(A[k:, k])))
            if A[piv, k] == 0:
                return np.zeros(n, dtype=np.float64)
            if piv != k:
                A[[k, piv]] = A[[piv, k]]
                b[[k, piv]] = b[[piv, k]]

            akk = A[k, k]
            for i in range(k + 1, n):
                factor = A[i, k] / akk
                if factor != 0.0:
                    A[i, k:] -= factor * A[k, k:]
                    b[i] -= factor * b[k]

        x = np.zeros(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            s = b[i] - np.dot(A[i, i + 1:], x[i + 1:])
            if A[i, i] == 0:
                return np.zeros(n, dtype=np.float64)
            x[i] = s / A[i, i]
        return x

    @staticmethod
    def _cheb_vec(t: float, m: int) -> np.ndarray:
        v = np.empty(m, dtype=np.float64)
        v[0] = 1.0
        if m == 1:
            return v
        v[1] = t
        for k in range(2, m):
            v[k] = 2.0 * t * v[k - 1] - v[k - 2]
        return v

    @staticmethod
    def _clenshaw_cheb(c: np.ndarray, t: float) -> float:
        n = c.shape[0]
        if n == 0:
            return 0.0
        b1 = 0.0
        b2 = 0.0
        for k in range(n - 1, 0, -1):
            b0 = 2.0 * t * b1 - b2 + float(c[k])
            b2 = b1
            b1 = b0
        return float(c[0]) + t * b1 - b2

    @staticmethod
    def _ridge_fit(X: np.ndarray, y: np.ndarray, lam_scale: float) -> np.ndarray:
        m = X.shape[1]
        Z = X.T @ X
        r = X.T @ y
        tr = float(np.trace(Z))
        lam = (lam_scale * tr / max(1, m)) if tr > 0 else lam_scale
        for i in range(m):
            Z[i, i] += lam
        return Assignment4._solve_gauss_pp(Z, r)

    @staticmethod
    def _safe_exp(u: float) -> float:
        if u > 700.0:
            return float(np.exp(700.0))
        if u < -700.0:
            return float(np.exp(-700.0))
        return float(np.exp(u))

    @staticmethod
    def _safe_exp_exp(u: float) -> float:
        # exp(exp(u)) — clamp u so exp(u) <= 700 (=> u <= ln(700) ~ 6.55)
        ln700 = 6.551080335
        if u > ln700:
            return float(np.exp(700.0))
        if u < -50.0:
            return float(np.exp(np.exp(-50.0)))
        return float(np.exp(np.exp(u)))

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        start = time.time()
        deadline = start + float(maxtime)
        hard_deadline = deadline + 4.9

        if a == b:
            y0 = float(f(a))
            return lambda x, y0=y0: y0

        # estimate one call cost (important for DELAYED)
        t0 = time.time()
        y_probe = float(f(0.5 * (a + b)))
        t1 = time.time()
        est_call = max(1e-4, t1 - t0)

        def can_start_call():
            now = time.time()
            return (now < deadline) and (hard_deadline - now >= est_call + 0.01)

        # choose stable local degree
        local_deg = min(10, int(d))
        m = local_deg + 1

        # number of segments: more segments when d is big (like your interpolation),
        # but bounded so we don't explode sampling
        segs = max(1, min(20, (int(d) + local_deg) // local_deg))
        edges = np.linspace(a, b, segs + 1, dtype=np.float64)

        # sample budget: enough for averaging noise per segment, but bounded
        # (calls dominate time; algebra is tiny)
        per_seg = max(20, 6 * m)
        max_total = min(2000, max(300, segs * per_seg))

        # collect samples per segment
        seg_x = [[] for _ in range(segs)]
        seg_y = [[] for _ in range(segs)]

        # seed each segment with a few grid points (good coverage)
        for s in range(segs):
            L = float(edges[s])
            R = float(edges[s + 1])
            xs = np.linspace(L, R, min(per_seg, 12), dtype=np.float64)
            for x in xs:
                if not can_start_call():
                    break
                t_call0 = time.time()
                y = float(f(float(x)))
                t_call1 = time.time()
                est_call = 0.85 * est_call + 0.15 * max(1e-4, t_call1 - t_call0)
                seg_x[s].append(float(x))
                seg_y[s].append(float(y))
            if not can_start_call():
                break

        # random fill until budget / time runs out
        total = sum(len(seg_x[s]) for s in range(segs))
        while total < max_total and can_start_call():
            x = random.uniform(a, b)
            s = int((x - a) * segs / (b - a))
            if s < 0:
                s = 0
            elif s >= segs:
                s = segs - 1

            t_call0 = time.time()
            y = float(f(x))
            t_call1 = time.time()
            est_call = 0.85 * est_call + 0.15 * max(1e-4, t_call1 - t_call0)

            seg_x[s].append(float(x))
            seg_y[s].append(float(y))
            total += 1

        # fit a small model per segment
        models = []
        eps = 1e-300

        for s in range(segs):
            xs = np.asarray(seg_x[s], dtype=np.float64)
            ys = np.asarray(seg_y[s], dtype=np.float64)

            L = float(edges[s])
            R = float(edges[s + 1])
            mid = 0.5 * (L + R)
            half = 0.5 * (R - L) if R != L else 1.0

            if xs.size < m:
                # fallback constant
                yconst = float(np.mean(ys)) if ys.size else float(y_probe)
                models.append(("const", yconst, L, R, mid, half, None))
                continue

            # normalize x -> t in [-1,1]
            ts = (xs - mid) / half
            ts = np.clip(ts, -1.0, 1.0)

            # build design matrix
            N = ts.shape[0]
            X = np.empty((N, m), dtype=np.float64)
            for i in range(N):
                X[i, :] = self._cheb_vec(float(ts[i]), m)

            # simple train/val split
            ntr = int(0.75 * N) if N >= 30 else N
            Xtr, ytr = X[:ntr], ys[:ntr]
            Xva, yva = X[ntr:], ys[ntr:]

            def mse(y_true, y_pred):
                if y_true.size == 0:
                    return 1e300
                return float(np.mean((y_true - y_pred) ** 2))

            # candidate A (y)
            c_y = self._ridge_fit(Xtr, ytr, lam_scale=1e-6)
            pred = (Xva @ c_y) if yva.size else (Xtr @ c_y)
            err_y = mse(yva if yva.size else ytr, pred)

            best_kind = "y"
            best_c = c_y
            best_err = err_y

            # candidate B (log y) if mostly positive
            if float(np.mean(ytr > 0.0)) > 0.9:
                ztr = np.log(np.maximum(ytr, eps))
                c_log = self._ridge_fit(Xtr, ztr, lam_scale=1e-6)
                if yva.size and float(np.mean(yva > 0.0)) > 0.9:
                    zpred = Xva @ c_log
                    ypred = np.exp(np.clip(zpred, -700.0, 700.0))
                    err = mse(yva, ypred)
                else:
                    zpred = Xtr @ c_log
                    ypred = np.exp(np.clip(zpred, -700.0, 700.0))
                    err = mse(ytr, ypred)
                if err < best_err:
                    best_kind = "log"
                    best_c = c_log
                    best_err = err

            # candidate C (log log y) if mostly > 1
            if float(np.mean(ytr > 1.0)) > 0.9:
                wtr = np.log(np.maximum(np.log(np.maximum(ytr, 1.0 + eps)), eps))
                c_ll = self._ridge_fit(Xtr, wtr, lam_scale=1e-6)
                if yva.size and float(np.mean(yva > 1.0)) > 0.9:
                    wpred = Xva @ c_ll
                    ypred = np.exp(np.clip(np.exp(np.clip(wpred, -50.0, 6.551080335)), -700.0, 700.0))
                    err = mse(yva, ypred)
                else:
                    wpred = Xtr @ c_ll
                    ypred = np.exp(np.clip(np.exp(np.clip(wpred, -50.0, 6.551080335)), -700.0, 700.0))
                    err = mse(ytr, ypred)
                if err < best_err:
                    best_kind = "loglog"
                    best_c = c_ll
                    best_err = err

            # refit chosen model on full segment data
            if best_kind == "y":
                c = self._ridge_fit(X, ys, lam_scale=1e-6)
            elif best_kind == "log":
                z = np.log(np.maximum(ys, eps))
                c = self._ridge_fit(X, z, lam_scale=1e-6)
            else:
                w = np.log(np.maximum(np.log(np.maximum(ys, 1.0 + eps)), eps))
                c = self._ridge_fit(X, w, lam_scale=1e-6)

            models.append((best_kind, c, L, R, mid, half, None))

        inv_span = segs / (b - a)

        def result(x):
            xv = float(x)
            si = int((xv - a) * inv_span)
            if si < 0:
                si = 0
            elif si >= segs:
                si = segs - 1

            kind, c, L, R, mid, half, _ = models[si]
            if kind == "const":
                return float(c)

            t = (xv - mid) / half
            if t < -1.0:
                t = -1.0
            elif t > 1.0:
                t = 1.0

            v = float(self._clenshaw_cheb(c, t))
            if kind == "y":
                return v
            if kind == "log":
                return self._safe_exp(v)
            return self._safe_exp_exp(v)

        return result

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):            
            self.assertNotEquals(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)

        
        



if __name__ == "__main__":
    unittest.main()
