"""
In this assignment you should implement an autodifferentiation framework from scratch
"""

import numpy as np
import time
import random

class Assignment6:
    class Variable:
        """
        Scalar reverse-mode autodiff Variable (float64).
        Supports: +, -, *, /, **, sin(), cos(), log10(), log()
        """

        __slots__ = ("x", "grad", "_prev", "_backward")

        def __init__(self, x):
            self.x = np.float64(x)
            self.grad = np.float64(0.0)
            self._prev = ()
            self._backward = lambda: None

        # ---------- helpers ----------
        @staticmethod
        def _as_var(v):
            return v if isinstance(v, Assignment6.Variable) else Assignment6.Variable(v)

        # ---------- core autodiff ----------
        def backward(self):
            # iterative topo sort (avoids recursion depth + a bit faster)
            topo = []
            visited = set()
            stack = [(self, 0)]
            while stack:
                node, state = stack.pop()
                nid = id(node)
                if state == 0:
                    if nid in visited:
                        continue
                    visited.add(nid)
                    stack.append((node, 1))
                    for p in node._prev:
                        stack.append((p, 0))
                else:
                    topo.append(node)

            # clear grads in this graph
            for v in topo:
                v.grad = np.float64(0.0)

            self.grad = np.float64(1.0)
            for v in reversed(topo):
                v._backward()

        # ---------- operator overloads ----------
        def __add__(self, other):
            other = Assignment6.Variable._as_var(other)
            out = Assignment6.Variable(self.x + other.x)
            out._prev = (self, other)

            def _backward():
                g = out.grad
                self.grad = np.float64(self.grad + g)
                other.grad = np.float64(other.grad + g)

            out._backward = _backward
            return out

        def __radd__(self, other):
            return self.__add__(other)

        def __sub__(self, other):
            other = Assignment6.Variable._as_var(other)
            out = Assignment6.Variable(self.x - other.x)
            out._prev = (self, other)

            def _backward():
                g = out.grad
                self.grad = np.float64(self.grad + g)
                other.grad = np.float64(other.grad - g)

            out._backward = _backward
            return out

        def __rsub__(self, other):
            other = Assignment6.Variable._as_var(other)
            return other.__sub__(self)

        def __mul__(self, other):
            other = Assignment6.Variable._as_var(other)
            out = Assignment6.Variable(self.x * other.x)
            out._prev = (self, other)

            def _backward():
                g = out.grad
                self.grad = np.float64(self.grad + other.x * g)
                other.grad = np.float64(other.grad + self.x * g)

            out._backward = _backward
            return out

        def __rmul__(self, other):
            return self.__mul__(other)

        def __truediv__(self, other):
            other = Assignment6.Variable._as_var(other)
            out = Assignment6.Variable(self.x / other.x)
            out._prev = (self, other)

            def _backward():
                g = out.grad
                inv = np.float64(1.0) / other.x
                self.grad = np.float64(self.grad + inv * g)
                other.grad = np.float64(other.grad - (self.x * inv * inv) * g)

            out._backward = _backward
            return out

        def __rtruediv__(self, other):
            other = Assignment6.Variable._as_var(other)
            return other.__truediv__(self)

        def __pow__(self, other):
            # supports scalar or Variable exponent
            if isinstance(other, Assignment6.Variable):
                base = self
                expv = other
                out_val = np.float64(base.x ** expv.x)
                out = Assignment6.Variable(out_val)
                out._prev = (base, expv)

                def _backward():
                    g = out.grad
                    # d/dx: out * exp / x  (more stable than exp*x**(exp-1))
                    base.grad = np.float64(base.grad + (out.x * expv.x / base.x) * g)
                    # d/dexp: out * ln(x)
                    expv.grad = np.float64(expv.grad + (out.x * np.log(base.x)) * g)

                out._backward = _backward
                return out
            else:
                p = np.float64(other)
                out_val = np.float64(self.x ** p)
                out = Assignment6.Variable(out_val)
                out._prev = (self,)

                def _backward():
                    g = out.grad
                    self.grad = np.float64(self.grad + (p * (self.x ** (p - 1.0))) * g)

                out._backward = _backward
                return out

        def __rpow__(self, other):
            # scalar ** Variable
            other = np.float64(other)
            out_val = np.float64(other ** self.x)
            out = Assignment6.Variable(out_val)
            out._prev = (self,)

            def _backward():
                g = out.grad
                self.grad = np.float64(self.grad + (out.x * np.log(other)) * g)

            out._backward = _backward
            return out

        def __neg__(self):
            out = Assignment6.Variable(-self.x)
            out._prev = (self,)

            def _backward():
                self.grad = np.float64(self.grad - out.grad)

            out._backward = _backward
            return out

        # ---------- math ops ----------
        def sin(self):
            out = Assignment6.Variable(np.sin(self.x))
            out._prev = (self,)

            def _backward():
                self.grad = np.float64(self.grad + (np.cos(self.x) * out.grad))

            out._backward = _backward
            return out

        def cos(self):
            out = Assignment6.Variable(np.cos(self.x))
            out._prev = (self,)

            def _backward():
                self.grad = np.float64(self.grad + (-np.sin(self.x) * out.grad))

            out._backward = _backward
            return out

        def log(self):
            # natural log
            out = Assignment6.Variable(np.log(self.x))
            out._prev = (self,)

            def _backward():
                self.grad = np.float64(self.grad + (out.grad / self.x))

            out._backward = _backward
            return out

        def log10(self):
            out = Assignment6.Variable(np.log10(self.x))
            out._prev = (self,)

            def _backward():
                self.grad = np.float64(self.grad + (out.grad / (self.x * np.log(10.0))))

            out._backward = _backward
            return out

    def __init__(self):
        pass

    def gradient(self, f, x) -> np.float64:
        """
        Reverse-mode scalar autodiff: O(#ops).
        Clears per-call graph grads to avoid accumulation across repeated calls.
        """
        vx = Assignment6.Variable(np.float64(x))
        y = f(vx)
        if not isinstance(y, Assignment6.Variable):
            y = Assignment6.Variable(y)
        y.backward()
        return np.float64(vx.grad)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment6(unittest.TestCase):

    def test_simple_function(self):
        f = lambda x: x**2
        x = np.float64(5)
        self.assertAlmostEqual(Assignment6().gradient(f, x), 2 * x)

    def test_complex_function(self):
        f = lambda x: x.sin() * x.cos()
        x = np.float64(5)
        self.assertAlmostEqual(Assignment6().gradient(f, x), np.cos(x)**2 - np.sin(x)**2)

    def test_function_with_multiple_operations(self):
        f = lambda x: (x**2 + x*2 + 1)**2
        x = np.float64(5)
        self.assertAlmostEqual(Assignment6().gradient(f, x), 2 * (x**2 + 2*x + 1) * (2*x + 2))

if __name__ == "__main__":
    unittest.main()
