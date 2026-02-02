"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np

class Assignment3:
    def __init__(self):
        """
        Initialization for pre-calculations.
        """
        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Calculates the definite integral using Gauss-Legendre Quadrature.
        """
        # Convert bounds to float64 for intermediate calculation stability
        lower, upper = np.float64(a), np.float64(b)
        
        # Obtain Legendre-Gauss nodes and weights
        nodes, weights = np.polynomial.legendre.leggauss(n)

        # Rescale nodes from [-1, 1] to [a, b]
        # formula: x = (b-a)/2 * node + (b+a)/2
        scaling_factor = 0.5 * (upper - lower)
        offset = 0.5 * (upper + lower)
        mapped_nodes = scaling_factor * nodes + offset
        
        # Evaluate function across all mapped nodes
        # List comprehension ensures the callable is handled correctly
        y_values = np.array([f(x) for x in mapped_nodes], dtype=np.float64)
        
        # The integral is the dot product of weights and values, scaled by (b-a)/2
        result = np.dot(weights, y_values) * scaling_factor
        
        return np.float32(result)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Computes the total absolute area between two functions by finding intersections.
        """
        # Define the difference function
        diff_func = lambda x: f1(x) - f2(x)
        
        # 1. Search for root intervals within the specified range [1, 100]
        search_range = np.linspace(1.0, 100.0, num=1000, dtype=np.float32)
        vals = np.array([diff_func(x) for x in search_range], dtype=np.float32)
        
        # Detect sign changes
        # np.sign(vals) gives -1, 0, or 1. np.diff finds where these change.
        sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
        
        intersections = []
        
        # 2. Refine roots using Bisection
        for idx in sign_changes:
            low, high = search_range[idx], search_range[idx + 1]
            
            # Standard bisection refinement
            for _ in range(25):
                mid = np.float32(0.5) * (low + high)
                if diff_func(low) * diff_func(mid) <= 0:
                    high = mid
                else:
                    low = mid
            
            root = np.float32(0.5) * (low + high)
            
            # Prevent adding duplicates that are too close together
            if not intersections or abs(root - intersections[-1]) > 1e-3:
                intersections.append(root)

        # Validation: Must have at least two points to enclose an area
        if len(intersections) < 2:
            return np.float32(np.nan)
            
        total_accumulated_area = np.float32(0.0)
        
        # 3. Sum the absolute area of each enclosed segment
        for j in range(len(intersections) - 1):
            left_bound = intersections[j]
            right_bound = intersections[j + 1]
            
            # Integrating the difference function per segment
            segment_val = self.integrate(diff_func, left_bound, right_bound, 64)
            total_accumulated_area += np.abs(segment_val)

        return np.float32(total_accumulated_area)
##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()