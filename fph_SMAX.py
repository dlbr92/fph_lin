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
df = pd.read_excel('batalh_ita.xlsx', sheet_name='Cortes_FPH_Linear_V_Faixa')
Caso = Newave('NEWAVE') #Lê os dados do Newave
uhe = Caso.hidr.get('Ita')          #5  2.14862*10E-5


plt.figure(figsize=(10, 6))

PCV =uhe['pol_cota_vol']
PVNJ = [2.65E+02,	9.05E-04,	-1.92E-08,	2.35E-13,	-1.19E-18]

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
    
def HL(Volume, Defluencia):
   return FCM(Volume)-FCJ(Defluencia)-uhe['perda_hid']

# Calcula S max da secante baseado no Pmax(Smax) da não linear
# Parâmetro fixo
q_max = 0
for i in range(uhe['num_conj_maq']):
     q_max=q_max+uhe['maq_por_conj'][i]*uhe['vaz_efet_conj'][i]
V_fixed = uhe['vol_max'] # Valor de volume máximo
# Eixo S (nível montante)
d_max = 56407.56  #Valor da defluência máxima

S_max =  np.inf
qued = HL(V_fixed, d_max)
p_max = uhe['prod_esp']*q_max*qued         
for _, row in df.iterrows():
    s = (row['Coef_Q'] * q_max +
         row['Coef_V'] * V_fixed +
         - p_max +
         row['Coef_Independente'])
    s = -(s/row['Coef_S'])
    S_max = np.minimum(S_max, s)


