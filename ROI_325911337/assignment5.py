import numpy as np
import time
from functionUtils import AbstractShape


class Assignment5:
    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # 1. Adaptive Area Computation (Shoelace)
    # ------------------------------------------------------------------
    def area(self, shape: AbstractShape, maxerr=0.001) -> np.float32:
        """
        Computes area using adaptive Shoelace formula refinement.
        Works for both Shape objects and contour functions.
        """

        # Handle input
        if hasattr(shape, "contour"):
            contour_func = shape.contour
        elif callable(shape):
            contour_func = shape
        else:
            raise ValueError("Input must be a Shape object or contour function")

        n = 100
        prev_area = None

        while True:
            pts = contour_func(n)
            pts = np.asarray(pts, dtype=np.float64)

            if len(pts) < 3:
                return np.float32(0.0)

            x = pts[:, 0]
            y = pts[:, 1]

            # Shoelace (vectorized)
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) -
                             np.dot(y, np.roll(x, -1)))

            # Convergence check
            if prev_area is not None:
                if area < 1e-12:
                    return np.float32(0.0)

                rel_err = abs(area - prev_area) / area
                if rel_err < maxerr:
                    return np.float32(area)

            if n > 20000:
                return np.float32(area)

            prev_area = area
            n *= 2

    # ------------------------------------------------------------------
    # 2. Shape Reconstruction
    # ------------------------------------------------------------------
    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Reconstruct shape from noisy samples.
        Adaptive smoothing based on radial spikiness metric.
        """

        start = time.time()
        buffer = 0.15
        points = []

        # --------------------------------------------------------------
        # 1. Collect Samples
        # --------------------------------------------------------------
        while time.time() - start < maxtime - buffer:
            try:
                points.append(sample())
            except Exception:
                break

            if len(points) >= 2000:
                break

        if len(points) < 3:
            return MyPolygonShape([])

        pts = np.asarray(points, dtype=np.float64)

        # --------------------------------------------------------------
        # 2. Sort by Angle Around Centroid
        # --------------------------------------------------------------
        centroid = pts.mean(axis=0)
        deltas = pts - centroid
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        order = np.argsort(angles)
        sorted_pts = pts[order]

        # --------------------------------------------------------------
        # 3. Spikiness Detection
        # --------------------------------------------------------------
        radii = np.linalg.norm(sorted_pts - centroid, axis=1)

        median_r = np.median(radii)
        max_r = np.max(radii)

        if median_r < 1e-12:
            spikiness = 0.0
        else:
            spikiness = max_r / median_r

        # --------------------------------------------------------------
        # 4. Adaptive Smoothing
        # --------------------------------------------------------------
        if spikiness <= 5.0 and len(sorted_pts) > 20:
            # Smooth radial distances (preserves corners better than XY smoothing)
            window = 15
            pad = window // 2

            radii_padded = np.pad(radii, (pad, pad), mode="wrap")
            kernel = np.ones(window) / window
            radii_smooth = np.convolve(radii_padded, kernel, mode="valid")

            theta = np.arctan2(sorted_pts[:, 1] - centroid[1],
                               sorted_pts[:, 0] - centroid[0])

            new_x = centroid[0] + radii_smooth * np.cos(theta)
            new_y = centroid[1] + radii_smooth * np.sin(theta)

            final_pts = np.column_stack((new_x, new_y))
        else:
            # Preserve spikes
            final_pts = sorted_pts

        return MyPolygonShape(final_pts)


# ----------------------------------------------------------------------
# Polygon Shape Implementation
# ----------------------------------------------------------------------
class MyPolygonShape(AbstractShape):
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)

    def contour(self, n: int):
        if len(self.points) == 0:
            return np.zeros((n, 2), dtype=np.float32)

        m = len(self.points)

        # Cyclic interpolation
        indices = np.linspace(0, m, n, endpoint=False)
        i0 = np.floor(indices).astype(int)
        i1 = (i0 + 1) % m
        alpha = (indices - i0)[:, None]

        p0 = self.points[i0]
        p1 = self.points[i1]

        interp = (1 - alpha) * p0 + alpha * p1
        return interp.astype(np.float32)

    def area(self) -> np.float32:
        if len(self.points) < 3:
            return np.float32(0.0)

        x = self.points[:, 0]
        y = self.points[:, 1]

        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) -
                         np.dot(y, np.roll(x, -1)))

        return np.float32(area)

###########################################################################
# tests (unchanged)
###########################################################################

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
