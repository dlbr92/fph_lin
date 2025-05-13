# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:11:40 2025

@author: dlbr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carrega a planilha
df = pd.read_excel('batalh_fph.xlsx', sheet_name='Cortes_FPH_Linear_V_Faixa')

# Parâmetro fixo
V_fixed = 1697.12  # hm³
Q_list = [0, 20, 50, 100, 130, 153]  # diferentes valores de vazão

# Eixo S (nível montante)
S_vals = np.linspace(0, 1382.2652399539948, 500)

plt.figure(figsize=(10, 6))

# Calcula e plota FPH para cada Q fixo
for Q_fixed in Q_list:
    FPH_vals = np.full(S_vals.shape, np.inf)
    for _, row in df.iterrows():
        fph = (row['Coef_Q'] * Q_fixed +
               row['Coef_V'] * V_fixed +
               row['Coef_S'] * S_vals +
               row['Coef_Independente'])
        FPH_vals = np.minimum(FPH_vals, fph)
    plt.plot(S_vals, FPH_vals, label=f'Q = {Q_fixed} m³/s')

# Estética do gráfico
plt.title(f'Curvas FPH vs. S (com V = {V_fixed} hm³)')
plt.xlabel('S m³/s')
plt.ylabel('FPH (MW)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()