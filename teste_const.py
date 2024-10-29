# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:18:08 2024

@author: David A
"""

def is_constant(polynomial):
    # Check if all coefficients except the first one are zero
    return all(coef == 0 for coef in polynomial[1:])

# Example usage
polynomial = [5, 0, 0, 0]  # This should be constant (only "a" term)
print(is_constant(polynomial))  # Output: True

polynomial = [5, 3, 0, 0]  # This is not constant
print(is_constant(polynomial))  # Output: False