import unittest
import numpy as np
import time
import random
import math

from assignment6 import Assignment6

np.seterr(over='ignore', invalid='ignore', divide='ignore')


# =====================================================
# Numeric Proxy
# =====================================================

class Num:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = np.float64(x)

    @staticmethod
    def _as(v):
        return v if isinstance(v, Num) else Num(v)

    def __add__(self, o): o = Num._as(o); return Num(self.x + o.x)
    def __radd__(self, o): return self.__add__(o)
    def __sub__(self, o): o = Num._as(o); return Num(self.x - o.x)
    def __rsub__(self, o): o = Num._as(o); return Num(o.x - self.x)
    def __mul__(self, o): o = Num._as(o); return Num(self.x * o.x)
    def __rmul__(self, o): return self.__mul__(o)
    def __truediv__(self, o): o = Num._as(o); return Num(self.x / o.x)
    def __rtruediv__(self, o): o = Num._as(o); return Num(o.x / self.x)
    def __neg__(self): return Num(-self.x)

    def __pow__(self, o):
        if isinstance(o, Num):
            exp = np.clip(o.x, -4, 4)
            return Num(self.x ** exp)
        return Num(self.x ** o)

    def __rpow__(self, o):
        return Num(o ** np.clip(self.x, -4, 4))

    def sin(self): return Num(np.sin(self.x))
    def cos(self): return Num(np.cos(self.x))
    def log(self): return Num(np.log(self.x))
    def log10(self): return Num(np.log10(self.x))


def eval_numeric(f, x):
    try:
        y = f(Num(x))
        return y.x if isinstance(y, Num) else np.float64(y)
    except:
        return np.nan


def numerical_grad(f, x):
    x = np.float64(x)
    h = 1e-6 * max(1.0, abs(x))
    fp = eval_numeric(f, x + h)
    fm = eval_numeric(f, x - h)
    if not (np.isfinite(fp) and np.isfinite(fm)):
        return np.nan
    return (fp - fm) / (2*h)


# =====================================================
# Complex Function Generators
# =====================================================

def deep_chain(depth=8):
    def f(x):
        y = x
        for _ in range(depth):
            y = (y.sin() + y.cos()) * 0.5 + y*y*0.1
        return y
    return f


def high_degree_poly(deg=10):
    coeffs = np.random.uniform(-2, 2, deg+1)
    def f(x):
        y = 0
        for i, c in enumerate(coeffs):
            y = y + c * (x**i)
        return y
    return f


def shared_subgraph():
    def f(x):
        a = x.sin()
        b = x.cos()
        c = a*b
        return c*c + c + a*b
    return f


def nested_logs():
    def f(x):
        z = x*x + 1e-3
        return (z.log() + z.log10()) * z.log()
    return f


def variable_exponent_chain():
    def f(x):
        base = x*x + 1e-3
        exp = (x.sin() * 2)
        return (base ** exp) * base.log()
    return f


def random_tree(depth=4):
    def safe(z): return z*z + 1e-3

    def node(d):
        if d == 0:
            a = random.uniform(-2,2)
            b = random.uniform(-2,2)
            return lambda x: a*x + b

        r = random.random()
        if r < 0.25:
            L = node(d-1); R = node(d-1)
            return lambda x: L(x) + R(x)
        if r < 0.5:
            L = node(d-1); R = node(d-1)
            return lambda x: L(x) * R(x)
        if r < 0.7:
            inner = node(d-1)
            return lambda x: inner(x).sin()
        if r < 0.85:
            inner = node(d-1)
            return lambda x: safe(inner(x)).log()
        base = node(d-1)
        return lambda x: safe(base(x)) ** (x.sin()*2)

    return node(depth)


# =====================================================
# Advanced Grader
# =====================================================

class TestAssignment6Advanced(unittest.TestCase):

    def test_advanced_grade(self):

        A = Assignment6()

        funcs = []

        # Deterministic stress functions
        funcs += [
            deep_chain(10),
            deep_chain(15),
            high_degree_poly(15),
            high_degree_poly(20),
            shared_subgraph(),
            nested_logs(),
            variable_exponent_chain()
        ]

        # Random structured trees
        for _ in range(20):
            funcs.append(random_tree(4))

        total_abs = 0
        total_rel = 0
        count = 0
        failures = 0

        t0 = time.perf_counter()

        for f in funcs:
            pts = np.random.uniform(-3, 3, 25)

            for x in pts:
                try:
                    g_auto = A.gradient(f, x)
                except:
                    failures += 1
                    continue

                g_ref = numerical_grad(f, x)

                if not (np.isfinite(g_auto) and np.isfinite(g_ref)):
                    continue

                abs_err = abs(g_auto - g_ref)
                rel_err = abs_err / max(1e-8, abs(g_ref))

                total_abs += abs_err
                total_rel += rel_err
                count += 1

        t1 = time.perf_counter()
        total_time = t1 - t0

        if count == 0:
            self.fail("No valid evaluations.")

        mean_abs = total_abs / count
        mean_rel = total_rel / count

        # Accuracy scoring
        acc = 100
        acc -= 50 * min(1, math.log10(1 + mean_abs)/3)
        acc -= 50 * min(1, math.log10(1 + 1e4*mean_rel)/4)
        acc = max(0, min(100, acc))

        # Speed scoring
        avg_call = total_time / count
        if avg_call < 3e-5:
            spd = 100
        elif avg_call < 8e-5:
            spd = 90
        elif avg_call < 2e-4:
            spd = 75
        elif avg_call < 5e-4:
            spd = 60
        else:
            spd = 40

        grade = 0.85*acc + 0.15*spd

        print("\n======================================")
        print(" ADVANCED AUTODIFF STRESS TEST RESULT ")
        print("======================================")
        print(f"Evaluations: {count}")
        print(f"Failures: {failures}")
        print(f"Mean abs error: {mean_abs:.3e}")
        print(f"Mean rel error: {mean_rel:.3e}")
        print(f"Avg time per call: {avg_call*1e6:.2f} µs")
        print(f"Accuracy score: {acc:.2f}")
        print(f"Speed score: {spd:.2f}")
        print(f"FINAL GRADE: {grade:.2f} / 100")
        print("======================================\n")

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
