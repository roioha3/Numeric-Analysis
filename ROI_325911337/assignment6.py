"""
In this assignment you should implement an autodifferentiation framework from scratch
"""
import numpy as np


class Assignment6:

    class Variable:
        """
        Forward-mode automatic differentiation variable.
        Holds:
            val : function value
            der : derivative w.r.t. input variable
        """

        __slots__ = ["val", "der"]

        def __init__(self, val, der=0.0):
            self.val = np.float64(val)
            self.der = np.float64(der)

        # ----------------------
        # Basic Arithmetic
        # ----------------------

        # (u + v)' = u' + v'
        def __add__(self, other):
            if isinstance(other, Assignment6.Variable):
                return Assignment6.Variable(
                    self.val + other.val,
                    self.der + other.der
                )
            return Assignment6.Variable(
                self.val + other,
                self.der
            )

        __radd__ = __add__

        # (u - v)' = u' - v'
        def __sub__(self, other):
            if isinstance(other, Assignment6.Variable):
                return Assignment6.Variable(
                    self.val - other.val,
                    self.der - other.der
                )
            return Assignment6.Variable(
                self.val - other,
                self.der
            )

        def __rsub__(self, other):
            return Assignment6.Variable(
                other - self.val,
                -self.der
            )

        # (u * v)' = u'v + uv'
        def __mul__(self, other):
            if isinstance(other, Assignment6.Variable):
                return Assignment6.Variable(
                    self.val * other.val,
                    self.der * other.val + self.val * other.der
                )
            return Assignment6.Variable(
                self.val * other,
                self.der * other
            )

        __rmul__ = __mul__

        # (u / v)' = (u'v - uv') / v²
        def __truediv__(self, other):
            if isinstance(other, Assignment6.Variable):
                return Assignment6.Variable(
                    self.val / other.val,
                    (self.der * other.val - self.val * other.der) / (other.val ** 2)
                )
            return Assignment6.Variable(
                self.val / other,
                self.der / other
            )

        def __rtruediv__(self, other):
            return Assignment6.Variable(
                other / self.val,
                -other * self.der / (self.val ** 2)
            )

        # ----------------------
        # Power
        # ----------------------

        # (u^v)' = u^v * (v' ln(u) + v u'/u)
        def __pow__(self, other):
            if isinstance(other, Assignment6.Variable):
                val = self.val ** other.val
                der = val * (
                    other.der * np.log(self.val)
                    + other.val * self.der / self.val
                )
                return Assignment6.Variable(val, der)

            # constant exponent
            val = self.val ** other
            der = other * (self.val ** (other - 1)) * self.der
            return Assignment6.Variable(val, der)

        def __rpow__(self, other):
            val = other ** self.val
            der = val * np.log(other) * self.der
            return Assignment6.Variable(val, der)

        # ----------------------
        # Elementary Functions
        # ----------------------

        def sin(self):
            return Assignment6.Variable(
                np.sin(self.val),
                np.cos(self.val) * self.der
            )

        def cos(self):
            return Assignment6.Variable(
                np.cos(self.val),
                -np.sin(self.val) * self.der
            )

        def log10(self):
            return Assignment6.Variable(
                np.log10(self.val),
                self.der / (self.val * np.log(10))
            )

        def log(self):
            return Assignment6.Variable(
                np.log(self.val),
                self.der / self.val
            )

        def exp(self):
            val = np.exp(self.val)
            return Assignment6.Variable(
                val,
                val * self.der
            )

    # ----------------------
    # Assignment Interface
    # ----------------------

    def __init__(self):
        pass

    def gradient(self, f, x) -> np.float64:
        """
        Compute derivative of scalar function f at scalar x.
        """
        var_x = self.Variable(x, 1.0)
        result = f(var_x)
        return np.float64(result.der)

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
