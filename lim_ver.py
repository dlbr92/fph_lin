# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:40:22 2025

@author: dlbr
"""
import pandas as pd
import numpy as np

def carregar_cortes(caminho_excel):
    """Lê os cortes da planilha."""
    return pd.read_excel(caminho_excel, sheet_name='Cortes_FPH_Linear_V_Faixa')

def fph(q, v, s, df):
    """Calcula o valor da função FPH (mínimo dos cortes) para (q, v, s)."""
    valores = df['Coef_Q'] * q + df['Coef_V'] * v + df['Coef_S'] * s
    return valores.min()

def encontrar_limite_superior_S(q, v, df, s_min=0, s_max=10000, tol=1e-3, max_iter=100):
    """
    Busca o maior valor de S tal que FPH(q, v, S) >= 0.
    Usa busca binária no intervalo [s_min, s_max].
    """
    baixo = s_min
    alto = s_max
    for _ in range(max_iter):
        meio = (baixo + alto) / 2
        potencia = fph(q, v, meio, df)
        if potencia >= 0:
            baixo = meio  # Tenta valores maiores
        else:
            alto = meio  # Tenta valores menores
        if abs(alto - baixo) < tol:
            break
    return baixo

# Exemplo de uso:
if __name__ == "__main__":
    caminho_excel = 'batalh_fph.xlsx'
    df_cortes = carregar_cortes(caminho_excel)

    q_fixo = 153.0
    v_fixo = 1697.12

    s_maximo = encontrar_limite_superior_S(q_fixo, v_fixo, df_cortes, s_min=0, s_max=50000)
    print(f"Maior valor de S com FPH >= 0 (q={q_fixo}, v={v_fixo}): {s_maximo:.3f}")