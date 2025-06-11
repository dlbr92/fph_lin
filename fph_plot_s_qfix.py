# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:11:40 2025

@author: dlbr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PySDDP.Pen import Newave

# Carrega a planilha
df = pd.read_excel('batalh_fph.xlsx', sheet_name='Cortes_FPH_Linear_V_Faixa')
Caso = Newave('NEWAVE') #Lê os dados do Newave
uhe = Caso.hidr.get('Batalha')          #5  2.14862*10E-5

# Parâmetro fixo
V_fixed = uhe['vol_max'] # hm³
Q_list = [ 153]  # diferentes valores de vazão

# Eixo S (nível montante)
S_vals = np.linspace(0, 1500, 500)

plt.figure(figsize=(10, 6))

PCV =uhe['pol_cota_vol']
PVNJ =uhe['pol_vaz_niv_jus']
def FCM( Volume):
        cota_montante = 0
        for i in range(len(PCV)):
            cota_montante += PCV[i]*Volume**i               
        return cota_montante
    
def FCJ( Defluencia):
        cota_jusante = 0
        for i in range(len(PVNJ)):
            cota_jusante += PVNJ[i]*Defluencia**i               
        return cota_jusante
def HL(Volume, Defluencia, w):
   return FCM(Volume)-FCJ(Defluencia)-uhe['perda_hid']

# Calcula e plota FPH para cada Q fixo
for Q_fixed in Q_list:
    FPH_vals = np.full(S_vals.shape, np.inf)
    FPH_vals_2 = list()
    for s in S_vals:
        qued = HL(V_fixed, s, Q_fixed)
        fph_2 = uhe['prod_esp']*Q_fixed*qued
        FPH_vals_2.append(fph_2)
   
    
    for _, row in df.iterrows():
        fph = (row['Coef_Q'] * Q_fixed +
               row['Coef_V'] * V_fixed +
               row['Coef_S'] * S_vals +
               row['Coef_Independente'])
        FPH_vals = np.minimum(FPH_vals, fph)
    plt.plot(S_vals, FPH_vals, label=f'Q = {Q_fixed} m³/s')
    plt.plot(S_vals, FPH_vals_2, '--', label=f'Exact Q = {Q_fixed} m³/s')

# Estética do gráfico
plt.title(f'Curvas FPH vs. S (com V = {V_fixed} hm³)')
plt.xlabel('S m³/s')
plt.ylabel('FPH (MW)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()