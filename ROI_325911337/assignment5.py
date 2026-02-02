import numpy as np
import time
from functionUtils import AbstractShape


class SplineShape(AbstractShape):
    def __init__(self, tck):
        self._tck = tck

    def contour(self, n: int):
        from scipy.interpolate import splev
        u = np.linspace(0.0, 1.0, num=n, endpoint=False)
        x, y = splev(u, self._tck)
        return np.stack([x, y], axis=1).astype(np.float32)

    def area(self):
        pts = self.contour(4096).astype(np.float64)
        x = pts[:, 0]
        y = pts[:, 1]
        return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


class PolygonShape(AbstractShape):
    def __init__(self, boundary_xy):
        self._P = np.asarray(boundary_xy, dtype=np.float64)

    def contour(self, n: int):
        P = self._P
        if len(P) == 0:
            return np.zeros((n, 2), dtype=np.float32)
        if len(P) == 1:
            return np.repeat(P.astype(np.float32), n, axis=0)
        idx = np.linspace(0, len(P), num=n, endpoint=False).astype(int) % len(P)
        return P[idx].astype(np.float32)

    def area(self):
        P = self._P
        if len(P) < 3:
            return 0.0
        x, y = P[:, 0], P[:, 1]
        return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


class CircleShape(AbstractShape):
    def __init__(self, cx, cy, r):
        self._cx = float(cx)
        self._cy = float(cy)
        self._r = float(max(r, 0.0))

    def contour(self, n: int):
        w = np.linspace(0, 2*np.pi, num=n, endpoint=False)
        x = self._cx + self._r*np.cos(w)
        y = self._cy + self._r*np.sin(w)
        return np.stack([x, y], axis=1).astype(np.float32)

    def area(self):
        return float(np.pi * self._r * self._r)


class Assignment5:
    def __init__(self):
        pass

    @staticmethod
    def _resample_closed_polyline(P, M=800):
        P = np.asarray(P, dtype=np.float64)
        n = len(P)
        if n < 2:
            return P

        Q = np.vstack([P, P[0]])
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        L = seg.sum()
        if L <= 1e-12:
            return P[:1]

        s = np.concatenate([[0.0], np.cumsum(seg)])
        targets = np.linspace(0.0, L, M + 1)[:-1]

        out = np.zeros((M, 2), dtype=np.float64)
        j = 0
        for i, t in enumerate(targets):
            while j + 1 < len(s) and s[j + 1] < t:
                j += 1
            if j >= len(seg):
                out[i] = Q[-1]
                continue
            dt = (t - s[j]) / max(seg[j], 1e-12)
            out[i] = (1 - dt) * Q[j] + dt * Q[j + 1]
        return out

    @staticmethod
    def _fit_periodic_spline(boundary_xy):
        from scipy.interpolate import splprep
        P = np.asarray(boundary_xy, dtype=np.float64)
        m = len(P)
        if m < 2:
            return None

        d = np.linalg.norm(np.diff(P, axis=0, append=P[:1]), axis=1)
        u = np.cumsum(d)
        u = np.insert(u, 0, 0.0)[:-1]
        if u[-1] <= 0:
            u = np.linspace(0, 1, m, endpoint=False)
        else:
            u = u / u[-1]

        x = P[:, 0]
        y = P[:, 1]
        k = min(3, m - 1)

        # light smoothing
        s = 0.00002 * m

        tck, _ = splprep([x, y], u=u, s=s, per=True, k=k)
        return tck

    # ---- circle fitting ----
    @staticmethod
    def _circle_fit_ls(P):
        P = np.asarray(P, dtype=np.float64)
        x = P[:, 0]
        y = P[:, 1]
        D = np.stack([x, y, np.ones_like(x)], axis=1)
        z = -(x*x + y*y)
        A, B, C = np.linalg.lstsq(D, z, rcond=None)[0]
        cx = -0.5 * A
        cy = -0.5 * B
        r2 = cx*cx + cy*cy - C
        r = np.sqrt(max(r2, 0.0))
        return float(cx), float(cy), float(r)

    @staticmethod
    def _circle_fit_geometric(P, iters=30):
        """
        Geometric (orthogonal) circle fit via Gauss–Newton + small-noise bias correction.
        Returns (cx, cy, r_corrected).
        """
        P = np.asarray(P, dtype=np.float64)
        x = P[:, 0]
        y = P[:, 1]

        # init
        cx, cy, r = Assignment5._circle_fit_ls(P)

        for _ in range(iters):
            dx = x - cx
            dy = y - cy
            di = np.sqrt(dx*dx + dy*dy) + 1e-12
            f = di - r

            J0 = -dx / di
            J1 = -dy / di
            J2 = -np.ones_like(di)

            A00 = np.dot(J0, J0)
            A01 = np.dot(J0, J1)
            A02 = np.dot(J0, J2)
            A11 = np.dot(J1, J1)
            A12 = np.dot(J1, J2)
            A22 = np.dot(J2, J2)

            b0 = -np.dot(J0, f)
            b1 = -np.dot(J1, f)
            b2 = -np.dot(J2, f)

            A = np.array([[A00, A01, A02],
                        [A01, A11, A12],
                        [A02, A12, A22]], dtype=np.float64)
            b = np.array([b0, b1, b2], dtype=np.float64)

            # a bit more damping than before (stabilizes with noisy data)
            lam = 1e-4
            A[0, 0] += lam
            A[1, 1] += lam
            A[2, 2] += lam

            delta = np.linalg.solve(A, b)

            cx += float(delta[0])
            cy += float(delta[1])
            r  += float(delta[2])

            if float(np.linalg.norm(delta)) < 1e-12:
                break

        # ---- bias correction ----
        dx = x - cx
        dy = y - cy
        di = np.sqrt(dx*dx + dy*dy) + 1e-12
        resid = di - r
        sigma = float(np.std(resid))  # estimates radial noise std

        # E[di] ≈ r + sigma^2/(2r)  ->  r ≈ r_hat - sigma^2/(2r_hat)
        r_corr = float(r - (sigma * sigma) / (2.0 * max(r, 1e-12)))
        r_corr = float(abs(r_corr))

        return float(cx), float(cy), r_corr

    @staticmethod
    def _looks_like_circle(P):
        P = np.asarray(P, dtype=np.float64)
        c = np.median(P, axis=0)
        Q = P - c
        cov = np.cov(Q.T)
        w = np.linalg.eigvalsh(cov)
        if w[0] <= 1e-12:
            return False
        eig_ratio = float(w[1] / w[0])

        r = np.linalg.norm(Q, axis=1)
        r_mean = float(np.mean(r))
        if r_mean <= 1e-12:
            return False
        r_cv = float(np.std(r) / r_mean)

        return (eig_ratio < 1.25) and (r_cv < 0.20)

    @staticmethod
    def _circumradius(pa, pb, pc):
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        s = 0.5 * (a + b + c)
        area2 = max(s * (s - a) * (s - b) * (s - c), 0.0)
        area = np.sqrt(area2)
        if area < 1e-14:
            return np.inf
        return (a * b * c) / (4.0 * area)

    @staticmethod
    def _alpha_shape_boundary_points(P, radius_max):
        """
        Compute alpha-shape boundary as an ordered polygon (points).
        radius_max is max allowed circumradius of kept triangles.
        Returns ordered boundary points (M,2) or None.
        """
        from scipy.spatial import Delaunay

        P = np.asarray(P, dtype=np.float64)
        if len(P) < 4:
            return None

        tri = Delaunay(P)
        edge_count = {}

        for ia, ib, ic in tri.simplices:
            pa, pb, pc = P[ia], P[ib], P[ic]
            R = Assignment5._circumradius(pa, pb, pc)
            if R <= radius_max:
                for e in ((ia, ib), (ib, ic), (ic, ia)):
                    e = tuple(sorted(e))
                    edge_count[e] = edge_count.get(e, 0) + 1

        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if len(boundary_edges) < 3:
            return None

        # adjacency
        adj = {}
        for a, b in boundary_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        # pick largest cycle component
        visited = set()
        best_cycle = None

        for start in list(adj.keys()):
            if start in visited:
                continue

            # collect component
            stack = [start]
            comp = []
            visited.add(start)
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in adj.get(v, []):
                    if u not in visited:
                        visited.add(u)
                        stack.append(u)

            # only a valid cycle if all degrees are 2
            if any(len(adj.get(v, [])) != 2 for v in comp):
                continue

            # order the cycle by walking
            s = min(comp)  # stable start
            cycle = [s]
            prev = None
            cur = s
            for _ in range(len(comp) + 5):
                nbrs = adj[cur]
                nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
                if nxt == s:
                    break
                cycle.append(nxt)
                prev, cur = cur, nxt

            if len(cycle) >= 3 and (best_cycle is None or len(cycle) > len(best_cycle)):
                best_cycle = cycle

        if best_cycle is None or len(best_cycle) < 3:
            return None

        return P[np.array(best_cycle, dtype=int)]

    @staticmethod
    def _convex_hull_points(P):
        from scipy.spatial import ConvexHull
        P = np.asarray(P, dtype=np.float64)
        if len(P) < 3:
            return P
        hull = ConvexHull(P)
        return P[hull.vertices]


    # ---- general boundary (kept as fallback for non-circles) ----
    @staticmethod
    def _robust_center(P):
        return np.median(P, axis=0)

    @staticmethod
    def _boundary_by_angle_bins(P, center, nbins=720, r_stat="median", min_per_bin=8):
        P = np.asarray(P, dtype=np.float64)
        c = np.asarray(center, dtype=np.float64)

        dx = P[:, 0] - c[0]
        dy = P[:, 1] - c[1]
        ang = np.arctan2(dy, dx)
        ang = (ang + 2 * np.pi) % (2 * np.pi)
        r = np.sqrt(dx * dx + dy * dy)

        bins = np.floor(ang / (2 * np.pi) * nbins).astype(int)
        bins = np.clip(bins, 0, nbins - 1)

        boundary = []
        for b in range(nbins):
            idx = np.where(bins == b)[0]
            if idx.size < min_per_bin:
                continue
            rr = r[idx]
            if r_stat == "median":
                rb = float(np.median(rr))
            elif r_stat == "trimmed_mean":
                lo, hi = np.quantile(rr, [0.2, 0.8])
                rr2 = rr[(rr >= lo) & (rr <= hi)]
                rb = float(np.mean(rr2)) if rr2.size else float(np.mean(rr))
            else:
                rb = float(np.mean(rr))

            theta = (b + 0.5) * (2 * np.pi / nbins)
            boundary.append([c[0] + rb * np.cos(theta), c[1] + rb * np.sin(theta)])

        return np.asarray(boundary, dtype=np.float64)

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area enclosed by a closed contour using adaptive
        Green's theorem (shoelace formula).

        Parameters
        ----------
        contour : callable
            Function contour(n) -> array of shape (n,2)
        maxerr : float
            Target absolute error

        Returns
        -------
        np.float32
            Estimated area
        """

        def polygon_area(pts):
            x = pts[:, 0]
            y = pts[:, 1]
            return 0.5 * abs(
                np.dot(x, np.roll(y, -1)) -
                np.dot(y, np.roll(x, -1))
            )

        n = 64
        A_prev = None

        while True:
            pts = contour(n).astype(np.float64)
            A = polygon_area(pts)

            if A_prev is not None and abs(A - A_prev) < maxerr:
                return np.float32(A)

            A_prev = A
            n *= 2

            # safety cap (never needed in practice, but prevents infinite loops)
            if n > 1_000_000:
                return np.float32(A)


    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        start = time.time()

        if maxtime >= 20:
            safety = 2.0
        elif maxtime >= 5:
            safety = 0.6
        else:
            safety = 0.15

        DEBUG = False
        MAX_STORE = 400_000 if maxtime >= 20 else 80_000

        pts = []
        while True:
            if time.time() - start >= maxtime - safety:
                break
            if len(pts) >= MAX_STORE:
                break
            x, y = sample()
            pts.append((x, y))

        pts = np.asarray(pts, dtype=np.float64)

        if DEBUG:
            print(f"[DEBUG] maxtime={maxtime}, safety={safety}")
            print(f"[DEBUG] stored pts: {len(pts)} in {time.time() - start:.3f}s (cap={MAX_STORE})")

        if len(pts) < 3:
            if DEBUG:
                print("[DEBUG] too few points -> PolygonShape fallback")
            return PolygonShape(pts)

        # ---- FAST PATH: geometric circle fit (fixes LS bias) ----
        SUB_C = 20000 if len(pts) > 20000 else len(pts)
        Pc = pts if len(pts) <= SUB_C else pts[np.random.choice(len(pts), SUB_C, replace=False)]

        if self._looks_like_circle(Pc):
            cx, cy, r = self._circle_fit_geometric(Pc, iters=30)
            if DEBUG:
                # quick check: recompute uncorrected mean radius for debug
                di = np.linalg.norm(Pc - np.array([cx, cy]), axis=1)
                print(f"[DEBUG] circle-geo(bias-corr) -> cx={cx:.6f}, cy={cy:.6f}, r_corr={r:.6f}, mean_dist={float(np.mean(di)):.6f}")
            return CircleShape(cx, cy, r)


        # ----------------------------
        # General fallback for polygons / arbitrary shapes:
        # alpha-shape (concave hull) -> PolygonShape
        # ----------------------------
        SUB = min(len(pts), 6000)
        P = pts if len(pts) <= SUB else pts[np.random.choice(len(pts), SUB, replace=False)]

        # scale from NN distances
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(P)
            d_nn, _ = tree.query(P, k=3)
            scale = float(np.median(d_nn[:, 1]))
        except Exception:
            scale = float(np.median(np.linalg.norm(P - np.roll(P, 1, axis=0), axis=1)))

        # try a few alpha levels (increasing permissiveness)
        radius_candidates = [2.0*scale, 3.0*scale, 5.0*scale, 8.0*scale, 12.0*scale, 20.0*scale]

        boundary = None
        for Rmax in radius_candidates:
            # keep runtime bounded
            if time.time() - start >= maxtime - safety:
                break
            boundary = self._alpha_shape_boundary_points(P, radius_max=Rmax)
            if boundary is not None and len(boundary) >= 3:
                break

        if boundary is None or len(boundary) < 3:
            # robust fallback: convex hull
            boundary = self._convex_hull_points(P)

        # For polygon-like shapes, returning PolygonShape preserves area best.
        # (Spline can shrink/expand and hurt area.)
        return PolygonShape(boundary)


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
