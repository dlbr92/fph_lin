# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:19:16 2024

@author: dlbr
"""

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def compute_concave_pwl_approximation(matrix):
    # Extract input and output columns from matrix
    x, y, z = matrix[:, 0], matrix[:, 1], matrix[:, 2]

    # Stack x, y to create 2D points for convex hull
    points = np.column_stack((x, y))

    # Compute convex hull
    hull = ConvexHull(points)

    # Extract the vertices of the hull
    hull_vertices = points[hull.vertices]
    
    # Find the concave region by inspecting the z values of the hull vertices
    concave_points = []
    
    for vertex in hull.vertices:
        if z[vertex] > np.mean(z):  # Assuming concave means higher z-values
            concave_points.append([x[vertex], y[vertex], z[vertex]])

    concave_points = np.array(concave_points)
    
    # Sort concave points based on x for PWL approximation
    sorted_concave_points = concave_points[concave_points[:, 0].argsort()]

    # Plot the concave region (optional)
    plt.figure(figsize=(8, 6))
    plt.tricontourf(x, y, z, levels=14, cmap='RdBu_r')
    plt.plot(hull_vertices[:, 0], hull_vertices[:, 1], 'go--', lw=2, label='Convex Hull')
    plt.plot(sorted_concave_points[:, 0], sorted_concave_points[:, 1], 'ro-', lw=2, label='Concave Region')
    plt.title('Concave Region with PWL Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='z')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate the linear equations for each segment of the concave region
    equations = []
    for i in range(len(sorted_concave_points) - 1):
        x1, y1, z1 = sorted_concave_points[i]
        x2, y2, z2 = sorted_concave_points[i+1]
        
        # Slope and intercept for 2D (x, y) plane equation y = m*x + b
        m = (y2 - y1) / (x2 - x1)  # slope
        b = y1 - m * x1            # intercept
        
        # Equation in 2D (x, y)
        equation_2d = f"y = {m}*x + {b}"
        equations.append((x1, x2, equation_2d))

        # Optional: Parametric equation in 3D (x, y, z)
        #equation_3d = f"x(t) = {(1 - t)}*{x1} + t*{x2}, y(t) = {(1 - t)}*{y1} + t*{y2}, z(t) = {(1 - t)}*{z1} + t*{z2}"
       # print(f"Segment from ({x1}, {y1}, {z1}) to ({x2}, {y2}, {z2}): {equation_2d} (2D), {equation_3d} (3D)")

    return sorted_concave_points, equations

# Example usage: Create a matrix of two inputs (x, y) and one output (z)
# The matrix is assumed to be a Nx3 array where the first two columns are inputs, and the third is output
matrix = np.random.rand(100, 3)  # Replace with your actual data

# Compute the concave PWL approximation and get the equations
concave_pwl_approximation, equations = compute_concave_pwl_approximation(matrix)

# Display the PWL points and corresponding equations
print("Concave PWL Points:\n", concave_pwl_approximation)
print("\nEquations for each segment of the concave region:")
for eq in equations:
    print(f"For x in [{eq[0]}, {eq[1]}]: {eq[2]}")
