# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:58:25 2024

@author: dlbr
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
# Generate a "reference" convex hull with a large number of points
def calculate_convex_hull_precision(num_points_list, reference_num_points):
    # Create a reference hull with many points to approximate the "true" area
    reference_points = np.random.rand(reference_num_points, 2)
    reference_hull = ConvexHull(reference_points)
    reference_area = reference_hull.area

    precision_errors = []
    for num_points in num_points_list:
        points = np.random.rand(num_points, 2)
        hull = ConvexHull(points)
        # Calculate the error as the absolute difference from the reference area
        error = np.abs(hull.area - reference_area) / reference_area
        precision_errors.append(error)
    
    return precision_errors

# Number of points to test and reference convex hull size
num_points_list = np.arange(10, 1010, 50)
reference_num_points = 5000

# Calculate precision errors
precision_errors = calculate_convex_hull_precision(num_points_list, reference_num_points)

# Plotting precision errors vs. number of points
plt.figure(figsize=(10, 6))
plt.plot(num_points_list, precision_errors, marker='o', label='Precision Error')
plt.title('Precision of Convex Hull Area Approximation vs. Number of Data Points')
plt.xlabel('Number of Data Points')
plt.ylabel('Relative Error (compared to reference area)')
plt.yscale('log')  # Log scale to better see precision changes
plt.grid(True)
plt.legend()
plt.show()