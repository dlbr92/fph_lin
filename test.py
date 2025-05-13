# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:41:23 2025

@author: dlbr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carrega a planilha
df = pd.read_excel('batalh_fph.xlsx', sheet_name='Cortes_FPH_Linear_V_Faixa')

# Parâmetros fixos
V_fixed = 1697.12  # hm³
S_list = [50, 100, 200, 300, 386]  # níveis montantes fixos

# Vetor de Q (vazão)
Q_vals = np.linspace(0, 153, 500)

plt.figure(figsize=(10, 6))

# Para cada S fixo, calcula o mínimo dos cortes (envoltória inferior)
for S_fixed in S_list:
    FPH_vals = np.full(Q_vals.shape, np.inf)
    for _, row in df.iterrows():
        fph = (row['Coef_Q'] * Q_vals +
               row['Coef_V'] * V_fixed +
               row['Coef_S'] * S_fixed)
        FPH_vals = np.minimum(FPH_vals, fph)
    plt.plot(Q_vals, FPH_vals, label=f'S = {S_fixed}')

# Estética do gráfico
plt.title(f'FPH vs Q para diferentes S (com V = {V_fixed} hm³)')
plt.xlabel('Q (m³/s)')
plt.ylabel('FPH (MW)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()