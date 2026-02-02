 # test_assignment3_grader.py
# A practical grader for Assignment3.integrate + Assignment3.areabetween
# - Scores 0..100
# - Precision weighted more than runtime
# - Enforces "call f at most n times"
# - Uses high-accuracy "ground truth" via mpmath (recommended) or scipy if available.
#
# Place this file next to Assignment3.py and run:
#   python test_assignment3_grader.py

import math
import time
import random
import numpy as np

# ----------------------------
# Import student's solution
# ----------------------------
from assignment3 import Assignment3


# ----------------------------
# Ground truth integrator
# ----------------------------
def _try_import_mpmath():
    try:
        import mpmath as mp
        return mp
    except Exception:
        return None

def _try_import_scipy_quad():
    try:
        from scipy.integrate import quad
        return quad
    except Exception:
        return None


_MP = _try_import_mpmath()
_QUAD = _try_import_scipy_quad()


def true_integral(f, a, b):
    """
    High-accuracy integral used as reference.
    Prefer mpmath.quad with high precision. Fallback to scipy.integrate.quad.
    """
    # Handle reversed ranges
    if b < a:
        return -true_integral(f, b, a)

    # mpmath: very high precision reference
    if _MP is not None:
        mp = _MP
        mp.mp.dps = 80  # high precision
        fa = float(a)
        fb = float(b)

        # mp.quad sometimes struggles on highly oscillatory intervals;
        # split the interval into chunks to stabilize.
        # More chunks for wide intervals.
        width = fb - fa
        chunks = 1
        if width > 5:
            chunks = 8
        if width > 30:
            chunks = 20

        pts = [fa + (width * i) / chunks for i in range(chunks + 1)]

        def f_mp(x):
            # f expects float -> float
            return mp.mpf(float(f(float(x))))

        total = mp.mpf("0")
        for i in range(chunks):
            total += mp.quad(f_mp, [pts[i], pts[i + 1]])
        return float(total)

    # scipy quad fallback
    if _QUAD is not None:
        quad = _QUAD
        val, _err = quad(lambda x: float(f(float(x))), float(a), float(b), limit=200)
        return float(val)

    raise RuntimeError(
        "Need either mpmath or scipy installed for ground-truth integration.\n"
        "Install with: pip install mpmath  (recommended)"
    )


def true_area_between(f1, f2, x_min=1.0, x_max=100.0):
    """
    Reference computation:
    - Find intersection points in [x_min, x_max] via dense sampling + sign-change bracketing + bisection.
    - Integrate abs(f1 - f2) piecewise between consecutive intersections.
    - If <2 intersections => NaN
    """
    def g(x):
        return float(f1(float(x)) - f2(float(x)))

    # Dense scan to find sign changes
    # Use more points to catch oscillations; still bounded.
    M = 6000
    xs = np.linspace(x_min, x_max, M, dtype=np.float64)
    ys = np.array([g(x) for x in xs], dtype=np.float64)

    # find intervals with sign changes or near-zero
    intervals = []
    eps0 = 1e-10
    for i in range(M - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (abs(y0) < eps0):
            intervals.append((xs[i] - (x_max - x_min) / M, xs[i] + (x_max - x_min) / M))
        if y0 == 0.0:
            continue
        if y0 * y1 < 0:
            intervals.append((xs[i], xs[i + 1]))

    # bisection to refine roots
    roots = []
    for (l, r) in intervals:
        l = max(x_min, float(l))
        r = min(x_max, float(r))
        if r <= l:
            continue
        gl = g(l)
        gr = g(r)

        # If not bracketed, skip (might be tangent root)
        if gl == 0.0:
            roots.append(l)
            continue
        if gr == 0.0:
            roots.append(r)
            continue
        if gl * gr > 0:
            continue

        for _ in range(80):
            m = 0.5 * (l + r)
            gm = g(m)
            if abs(gm) < 1e-12:
                l = r = m
                break
            if gl * gm <= 0:
                r = m
                gr = gm
            else:
                l = m
                gl = gm
        roots.append(0.5 * (l + r))

    # Deduplicate & sort roots
    roots = sorted(roots)
    dedup = []
    for x in roots:
        if not dedup or abs(x - dedup[-1]) > 1e-4:
            dedup.append(x)
    roots = dedup

    if len(roots) < 2:
        return float("nan")

    # Area = sum over consecutive intervals of integral(|g|)
    area = 0.0
    for i in range(len(roots) - 1):
        a, b = roots[i], roots[i + 1]
        if b <= a:
            continue

        def abs_g(x):
            return abs(f1(float(x)) - f2(float(x)))

        seg = true_integral(abs_g, a, b)
        area += abs(seg)

    return float(area)


# ----------------------------
# Call-counting wrapper
# ----------------------------
class CountCalls:
    def __init__(self, f):
        self.f = f
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.f(x)


# ----------------------------
# Test function generators
# ----------------------------
def poly_func(deg=5, coeff_scale=1.0):
    coeffs = [random.uniform(-coeff_scale, coeff_scale) for _ in range(deg + 1)]
    p = np.poly1d(coeffs)
    return lambda x: float(p(x))

def trig_mix():
    A = random.uniform(0.1, 2.0)
    B = random.uniform(0.1, 2.0)
    w1 = random.uniform(0.5, 12.0)
    w2 = random.uniform(0.5, 12.0)
    phi1 = random.uniform(0, 2*math.pi)
    phi2 = random.uniform(0, 2*math.pi)
    return lambda x: float(A*math.sin(w1*x + phi1) + B*math.cos(w2*x + phi2))

def strong_oscillation_like():
    # Similar vibe to "strong_oscilations": large swings + oscillations
    # Keep finite and integrable on tested ranges.
    w = random.uniform(20.0, 120.0)
    scale = random.uniform(0.5, 5.0)
    return lambda x: float(scale * math.sin(w*x) * math.exp(0.15*x))

def exp_poly_mix():
    a = random.uniform(-0.4, 0.4)
    b = random.uniform(-2.0, 2.0)
    p = poly_func(deg=random.randint(2, 6), coeff_scale=1.5)
    return lambda x: float(math.exp(a*x) * p(x) + b)

def rational_like():
    # Avoid singularities by shifting
    c = random.uniform(1.0, 5.0)
    d = random.uniform(0.2, 2.0)
    return lambda x: float((x*x + 1.0) / (x + c) + d*math.sin(2.0*x))

def piecewise_smooth():
    # smooth-ish piecewise using tanh transition
    k = random.uniform(2.0, 10.0)
    t = random.uniform(-1.0, 1.0)
    p1 = poly_func(deg=3, coeff_scale=1.0)
    p2 = poly_func(deg=3, coeff_scale=1.0)
    return lambda x: float(0.5*(p1(x)+p2(x)) + 0.5*(p1(x)-p2(x))*math.tanh(k*(x-t)))

def random_function_family():
    gens = [poly_func, trig_mix, exp_poly_mix, rational_like, piecewise_smooth, strong_oscillation_like]
    g = random.choice(gens)
    return g()


# ----------------------------
# Scoring utilities
# ----------------------------
def rel_error(pred, true, eps=1e-30):
    # robust relative error
    denom = max(abs(true), eps)
    return abs(pred - true) / denom

def score_from_error(err, good=1e-3, ok=1e-2, bad=5e-2):
    """
    Map error -> score in [0,1].
    - <= good : ~1
    - ok      : ~0.7
    - bad     : ~0.2
    - worse   : decays to 0
    """
    if not np.isfinite(err):
        return 0.0
    if err <= good:
        return 1.0
    if err <= ok:
        # 1 -> 0.7 linearly
        return 1.0 - (err - good) * (0.3 / (ok - good))
    if err <= bad:
        # 0.7 -> 0.2 linearly
        return 0.7 - (err - ok) * (0.5 / (bad - ok))
    # beyond bad: exponential decay
    return float(0.2 * math.exp(-(err - bad) / (bad + 1e-12)))

def score_from_time(elapsed, budget):
    """
    Time score in [0,1].
    full score if <= budget, then decays.
    """
    if elapsed <= budget:
        return 1.0
    # soft penalty
    return float(math.exp(-(elapsed - budget) / max(budget, 1e-9)))


# ----------------------------
# Integrate tests
# ----------------------------
def run_integrate_tests(ass3, seed=0, num_tests=40):
    random.seed(seed)
    np.random.seed(seed)

    # (a,b,n) choices
    ranges = [
        (-1.0, 1.0, 10),
        (0.0, 2.0, 20),
        (0.1, 3.5, 30),
        (0.09, 10.0, 20),   # "hard-ish" style
        (-2.0, 5.0, 60),
        (0.0, 30.0, 100),
    ]

    results = []
    for i in range(num_tests):
        f = random_function_family()
        a, b, n = random.choice(ranges)

        wrapped = CountCalls(f)

        # run student
        t0 = time.perf_counter()
        try:
            pred = ass3.integrate(wrapped, float(a), float(b), int(n))
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append({
                "name": f"integrate#{i}",
                "ok": False,
                "error": float("inf"),
                "time": elapsed,
                "calls": wrapped.calls,
                "note": f"exception: {e}",
            })
            continue
        elapsed = time.perf_counter() - t0

        # checks
        dtype_ok = isinstance(pred, np.float32) or (hasattr(pred, "dtype") and pred.dtype == np.float32)
        calls_ok = wrapped.calls <= n

        # reference
        try:
            true = true_integral(f, a, b)
            err = rel_error(float(pred), true)
        except Exception as e:
            true = float("nan")
            err = float("inf")

        ok = dtype_ok and calls_ok and np.isfinite(float(pred))
        note_parts = []
        if not dtype_ok:
            note_parts.append("dtype not float32")
        if not calls_ok:
            note_parts.append(f"too many calls ({wrapped.calls}>{n})")
        note = ", ".join(note_parts) if note_parts else ""

        results.append({
            "name": f"integrate#{i}",
            "ok": ok,
            "error": err,
            "time": elapsed,
            "calls": wrapped.calls,
            "note": note,
        })

    return results


# ----------------------------
# areabetween tests
# ----------------------------
def make_intersecting_pair():
    """
    Create two functions likely to have multiple intersections in [1,100].
    We use combinations of polynomials + trig to increase crossings.
    """
    base1 = poly_func(deg=random.randint(2, 5), coeff_scale=0.01)
    base2 = poly_func(deg=random.randint(2, 5), coeff_scale=0.01)

    trig1 = trig_mix()
    trig2 = trig_mix()

    s1 = random.uniform(0.2, 2.0)
    s2 = random.uniform(0.2, 2.0)
    t1 = random.uniform(0.1, 1.0)
    t2 = random.uniform(0.1, 1.0)

    f1 = lambda x: float(s1*base1(x) + t1*trig1(x))
    f2 = lambda x: float(s2*base2(x) + t2*trig2(x))
    return f1, f2


def run_areabetween_tests(ass3, seed=0, num_tests=30):
    random.seed(seed + 12345)
    np.random.seed(seed + 12345)

    results = []
    for i in range(num_tests):
        f1, f2 = make_intersecting_pair()

        # student
        t0 = time.perf_counter()
        try:
            pred = ass3.areabetween(f1, f2)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append({
                "name": f"areabetween#{i}",
                "ok": False,
                "error": float("inf"),
                "time": elapsed,
                "note": f"exception: {e}",
            })
            continue
        elapsed = time.perf_counter() - t0

        # reference
        try:
            true = true_area_between(f1, f2, 1.0, 100.0)
        except Exception:
            true = float("nan")

        # Expected behavior: if <2 intersections => NaN
        if not np.isfinite(true):
            ok = (isinstance(pred, np.float32) or (hasattr(pred, "dtype") and pred.dtype == np.float32)) and (not np.isfinite(float(pred)))
            err = 0.0 if ok else float("inf")
            results.append({
                "name": f"areabetween#{i}",
                "ok": ok,
                "error": err,
                "time": elapsed,
                "note": "expected NaN (less than 2 intersections)" if ok else "should return NaN",
            })
            continue

        # Otherwise must be non-negative area and float32
        dtype_ok = isinstance(pred, np.float32) or (hasattr(pred, "dtype") and pred.dtype == np.float32)
        nonneg_ok = float(pred) >= -1e-6  # tolerate tiny negative from float error
        finite_ok = np.isfinite(float(pred))

        err = rel_error(float(pred), true)
        ok = dtype_ok and nonneg_ok and finite_ok

        note_parts = []
        if not dtype_ok:
            note_parts.append("dtype not float32")
        if not nonneg_ok:
            note_parts.append("negative area")
        if not finite_ok:
            note_parts.append("not finite")
        note = ", ".join(note_parts) if note_parts else ""

        results.append({
            "name": f"areabetween#{i}",
            "ok": ok,
            "error": err,
            "time": elapsed,
            "note": note,
        })

    return results


# ----------------------------
# Final grading
# ----------------------------
def grade_results(integrate_res, area_res,
                  precision_weight=0.75, time_weight=0.25,
                  integrate_time_budget=0.015, area_time_budget=0.12):
    """
    Combine precision and time into 0..100.
    Budgets are per-test (seconds), tweak to your machine.
    """

    def score_test(r, budget):
        # If basic correctness fails, give 0
        if not r["ok"]:
            return 0.0, 0.0, 0.0

        pe = score_from_error(r["error"])
        ts = score_from_time(r["time"], budget)
        total = precision_weight * pe + time_weight * ts
        return total, pe, ts

    # integrate
    integ_scores = [score_test(r, integrate_time_budget) for r in integrate_res]
    area_scores = [score_test(r, area_time_budget) for r in area_res]

    # average
    integ_avg = sum(s[0] for s in integ_scores) / max(len(integ_scores), 1)
    area_avg = sum(s[0] for s in area_scores) / max(len(area_scores), 1)

    # overall: 50/50 between tasks (you can change)
    overall = 0.5 * integ_avg + 0.5 * area_avg

    return {
        "overall_0_100": 100.0 * overall,
        "integrate_0_100": 100.0 * integ_avg,
        "areabetween_0_100": 100.0 * area_avg,
        "integrate_details": integ_scores,
        "areabetween_details": area_scores,
    }


def print_report(integrate_res, area_res, grade):
    def summarize(res):
        ok = sum(1 for r in res if r["ok"])
        total = len(res)
        worst = sorted([r for r in res if r["ok"]], key=lambda x: x["error"], reverse=True)[:5]
        return ok, total, worst

    ok_i, total_i, worst_i = summarize(integrate_res)
    ok_a, total_a, worst_a = summarize(area_res)

    print("\n==================== Assignment 3 Local Grader ====================")
    print(f"Overall grade:       {grade['overall_0_100']:.2f} / 100")
    print(f" integrate grade:    {grade['integrate_0_100']:.2f} / 100   (ok {ok_i}/{total_i})")
    print(f" areabetween grade:  {grade['areabetween_0_100']:.2f} / 100   (ok {ok_a}/{total_a})")

    print("\n--- Worst integrate errors (among passing correctness) ---")
    for r in worst_i:
        print(f"{r['name']}: rel_err={r['error']:.3e}, time={r['time']*1e3:.2f}ms, calls={r['calls']}, note={r['note']}")

    print("\n--- Worst areabetween errors (among passing correctness) ---")
    for r in worst_a:
        print(f"{r['name']}: rel_err={r['error']:.3e}, time={r['time']*1e3:.2f}ms, note={r['note']}")

    # Show some failing cases
    bad_i = [r for r in integrate_res if not r["ok"]][:8]
    bad_a = [r for r in area_res if not r["ok"]][:8]

    if bad_i:
        print("\n--- Some integrate failures ---")
        for r in bad_i:
            print(f"{r['name']}: time={r['time']*1e3:.2f}ms, calls={r['calls']}, note={r['note']}")

    if bad_a:
        print("\n--- Some areabetween failures ---")
        for r in bad_a:
            print(f"{r['name']}: time={r['time']*1e3:.2f}ms, note={r['note']}")

    print("===================================================================\n")


def main():
    # Tweaks
    SEED = 0
    NUM_INTEGRATE_TESTS = 50
    NUM_AREABETWEEN_TESTS = 35

    ass3 = Assignment3()

    # sanity: dtype test similar to theirs
    f_poly = np.poly1d([-1, 0, 1])
    out = ass3.integrate(f_poly, -1.0, 1.0, 10)
    if not (isinstance(out, np.float32) or (hasattr(out, "dtype") and out.dtype == np.float32)):
        print("WARNING: integrate output is not float32 in quick sanity test.")

    integrate_res = run_integrate_tests(ass3, seed=SEED, num_tests=NUM_INTEGRATE_TESTS)
    area_res = run_areabetween_tests(ass3, seed=SEED, num_tests=NUM_AREABETWEEN_TESTS)

    grade = grade_results(integrate_res, area_res)
    print_report(integrate_res, area_res, grade)


if __name__ == "__main__":
    main()
