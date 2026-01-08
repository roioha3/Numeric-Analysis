"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape

class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, center=None, coeffs=None, order=0, r_clip=None, fallback_points=None):
        super(MyShape, self).__init__()
        self._center = center
        self._coeffs = coeffs
        self._order = int(order)
        self._r_clip = r_clip  # (rmin, rmax) in float64
        self._fallback_points = fallback_points

    def _r_from_theta(self, theta: np.ndarray) -> np.ndarray:
        c = self._coeffs
        if c is None:
            return None
        r = np.full_like(theta, c[0], dtype=np.float64)
        idx = 1
        for k in range(1, self._order + 1):
            r += c[idx] * np.cos(k * theta); idx += 1
            r += c[idx] * np.sin(k * theta); idx += 1
        if self._r_clip is not None:
            rmin, rmax = self._r_clip
            r = np.clip(r, rmin, rmax)
        r = np.maximum(r, 0.0)
        return r

    @staticmethod
    def _area_points(pts: np.ndarray) -> float:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            return 0.0

        # try given order
        x = pts[:, 0]
        y = pts[:, 1]
        a1 = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        a1 = abs(a1) if np.isfinite(a1) else 0.0

        # try angle-sorted order (robust if points not ordered)
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        p2 = pts[np.argsort(ang)]
        x2 = p2[:, 0]
        y2 = p2[:, 1]
        a2 = 0.5 * (np.dot(x2, np.roll(y2, -1)) - np.dot(y2, np.roll(x2, -1)))
        a2 = abs(a2) if np.isfinite(a2) else 0.0

        if a1 <= 0:
            return a2
        if a2 <= 0:
            return a1
        return min(a1, a2)

    def contour(self, n: int):
        n = int(n)
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        if self._center is None or self._coeffs is None:
            pts = self._fallback_points
            if pts is None or len(pts) == 0:
                return np.zeros((n, 2), dtype=np.float32)
            pts = np.asarray(pts, dtype=np.float64)
            if pts.shape[0] == n:
                return pts.astype(np.float32)
            idx = np.linspace(0, pts.shape[0] - 1, n, dtype=np.int64)
            return pts[idx].astype(np.float32)

        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
        r = self._r_from_theta(theta)
        x = self._center[0] + r * np.cos(theta)
        y = self._center[1] + r * np.sin(theta)
        return np.stack([x, y], axis=1).astype(np.float32)

    def area(self) -> np.float32:
        pts = self.contour(4096)
        return np.float32(self._area_points(pts))


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        maxerr = float(maxerr)

        def area_points(pts: np.ndarray) -> float:
            pts = np.asarray(pts, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
                return 0.0

            x = pts[:, 0]
            y = pts[:, 1]
            a1 = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            a1 = abs(a1) if np.isfinite(a1) else 0.0

            c = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            p2 = pts[np.argsort(ang)]
            x2 = p2[:, 0]
            y2 = p2[:, 1]
            a2 = 0.5 * (np.dot(x2, np.roll(y2, -1)) - np.dot(y2, np.roll(x2, -1)))
            a2 = abs(a2) if np.isfinite(a2) else 0.0

            if a1 <= 0:
                return a2
            if a2 <= 0:
                return a1
            return min(a1, a2)

        n = 64
        prev = None
        best = 0.0
        while True:
            a = area_points(contour(n))
            best = a
            if prev is not None and abs(a - prev) <= maxerr:
                return np.float32(a)
            prev = a
            if n >= 10000:
                return np.float32(best)
            n *= 2

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """
        t0 = time.time()
        deadline = t0 + float(maxtime)
        soft_deadline = deadline - 0.05

        pts = []
        max_pts = 12000

        while len(pts) < max_pts and time.time() < soft_deadline:
            x, y = sample()
            if np.isfinite(x) and np.isfinite(y):
                pts.append((float(x), float(y)))

        if len(pts) < 30:
            return MyShape(fallback_points=np.array(pts, dtype=np.float32) if len(pts) else None)

        P = np.asarray(pts, dtype=np.float64)

        # robust center
        cx = np.median(P[:, 0])
        cy = np.median(P[:, 1])

        X = P[:, 0] - cx
        Y = P[:, 1] - cy
        theta = np.arctan2(Y, X)
        r = np.hypot(X, Y)

        # drop extreme outliers before binning
        r_lo = np.quantile(r, 0.02)
        r_hi = np.quantile(r, 0.98)
        keep = (r >= r_lo) & (r <= r_hi)
        theta = theta[keep]
        r = r[keep]

        # bin-by-angle median radius (robust)
        B = 360
        edges = np.linspace(-np.pi, np.pi, B + 1, dtype=np.float64)
        bin_id = np.clip(np.searchsorted(edges, theta, side="right") - 1, 0, B - 1)

        r_med = np.full(B, np.nan, dtype=np.float64)
        counts = np.zeros(B, dtype=np.int64)
        for i in range(B):
            sel = (bin_id == i)
            if np.any(sel):
                counts[i] = int(np.count_nonzero(sel))
                r_med[i] = np.median(r[sel])

        # fill empty bins by nearest neighbor forward/backward
        if np.any(np.isnan(r_med)):
            idx_valid = np.where(~np.isnan(r_med))[0]
            if idx_valid.size == 0:
                return MyShape(fallback_points=P.astype(np.float32))
            for i in range(B):
                if np.isnan(r_med[i]):
                    j = idx_valid[np.argmin(np.minimum((idx_valid - i) % B, (i - idx_valid) % B))]
                    r_med[i] = r_med[j]

        th_grid = (edges[:-1] + edges[1:]) * 0.5

        # Fourier order (keep small for stability/speed)
        order = 10
        if len(P) > 5000:
            order = 12

        # ridge-stabilized least squares via augmentation
        n = th_grid.shape[0]
        cols = 1 + 2 * order
        A = np.empty((n, cols), dtype=np.float64)
        A[:, 0] = 1.0
        idx = 1
        for k in range(1, order + 1):
            A[:, idx] = np.cos(k * th_grid); idx += 1
            A[:, idx] = np.sin(k * th_grid); idx += 1

        lam = 1e-2
        I = np.eye(cols, dtype=np.float64)
        I[0, 0] = 0.0
        A_aug = np.vstack([A, np.sqrt(lam) * I])
        b_aug = np.concatenate([r_med, np.zeros(cols, dtype=np.float64)])
        coeffs, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)

        # clamp radii during generation
        rmin = float(np.quantile(r_med, 0.01) * 0.85)
        rmax = float(np.quantile(r_med, 0.99) * 1.15)
        rmin = max(0.0, rmin)
        rmax = max(rmin + 1e-6, rmax)

        return MyShape(
            center=(float(cx), float(cy)),
            coeffs=coeffs.astype(np.float64),
            order=order,
            r_clip=(rmin, rmax),
            fallback_points=P.astype(np.float32),
        )



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
