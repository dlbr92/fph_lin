# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:56:14 2025

@author: dlbr
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Load the sheet containing the cuts for the FPH approximation

df = pd.read_excel('batalh_fph.xlsx', sheet_name='Cortes_FPH_Linear_V_Faixa')

# Display the first few rows to understand the structure
df.head()

# Fix the volume
V_fixed = 1697.12  # in hm³

# Create meshgrid for Q and S
Q_vals = np.linspace(0, 153, 100)
S_vals = np.linspace(0, 1382.2652399539948, 100)
Q, S = np.meshgrid(Q_vals, S_vals)

# Initialize matrix for max FPH value from cuts
FPH = np.full(Q.shape, np.inf)

# Apply all cuts and keep the maximum value for each (Q, S) pair
for _, row in df.iterrows():
    fph_val = (row['Coef_Q'] * Q +
               row['Coef_V'] * V_fixed +
               row['Coef_S'] * S +
               row['Coef_Independente'])
    FPH = np.minimum(FPH, fph_val)

# Plotting the FPH surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Q, S, FPH, cmap='viridis')
ax.set_title('FPH aproximada por partes para V = 1697.12 hm³ - Batalha', fontsize=12)
ax.set_xlabel('Q (m³/s)', fontsize=12)
ax.set_ylabel('S(m³/s)', fontsize=12)
ax.set_zlabel('FPH (MW)', fontsize=12)
plt.tight_layout()
plt.show()


# Plot da curva de nível
plt.figure(figsize=(10, 7))
contour = plt.contourf(Q, S, FPH, levels=20, cmap='viridis')
cbar = plt.colorbar(contour)
cbar.set_label('FPH (MW)')

plt.title('Curvas de Nível da FPH (V = 1697,12 hm³)')
plt.xlabel('Q (m³/s)')
plt.ylabel('S (m³/s)')
plt.grid(True)
plt.tight_layout()
plt.show()

