# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:13:53 2025

@author: dlbr
"""

import pandas as pd
import numpy as np
from PySDDP.Pen import Newave

def carregar_cortes(caminho_csv):
    """Lê os cortes da planilha CSV."""
    return pd.read_csv(caminho_csv, delimiter=';')

def fph(q, v, p, df):
    """Calcula o valor da função FPH (mínimo dos cortes) para (q, v, s)."""
    df_validos = df[
        (df['Coef_S'] != 0) & (~df['Coef_S'].isna()) &
        (~df['Coef_Q'].isna()) & (~df['Coef_V'].isna()) & (~df['Coef_D'].isna())
    ]
    if df_validos.empty:
        return np.nan
    valores = df_validos['Coef_Q'] * q + df_validos['Coef_V'] * v - p + df_validos['Coef_D']
    valores_validos = -(valores / df_validos['Coef_S'])
    return np.nanmin(valores_validos)

def encontrar_limite_superior_S(q, v, p_fixo, df):
    """
    Busca o maior valor de S tal que FPH(q, v, S) >= 0 usando busca binária.
    """
    s_max = fph(q, v, p_fixo, df)
    if np.isinf(s_max) or np.isnan(s_max):
        return None
    return s_max

def FCM(Volume, PCV):
    cota_montante = 0
    for i in range(len(PCV)):
        cota_montante += PCV[i]*Volume**i               
    return cota_montante

def FCJ(Defluencia, PVNJ):
    cota_jusante = 0
    for i in range(len(PVNJ)):
        cota_jusante += PVNJ[i]*Defluencia**i               
    return cota_jusante

def HL(Volume, Defluencia, h_loss, PCV, PVNJ):
    return FCM(Volume, PCV)-FCJ(Defluencia, PVNJ) - h_loss

def main():
    caminho_csv = 'fphs_cortes.csv'  # Substitua pelo caminho correto no seu PC
    df = carregar_cortes(caminho_csv)

    # Renomeia colunas
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={
        "ID da Usina": "ID",
        "Coeficiente de Volume": "Coef_V",
        "Coeficiente de Turbinamento": "Coef_Q",
        "Coeficiente de Vertimento": "Coef_S",
        "Termo Independente": "Coef_D",
    })

    # Conversão segura para float nas colunas numéricas
    for col in ['Coef_Q', 'Coef_V', 'Coef_S']:
        df[col] = df[col].astype(str).str.replace("\u202f", "").str.replace(' ', '').str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Coef_D'] = pd.to_numeric(df['Coef_D'], errors='coerce')

    # Caminho para o arquivo CSV
    caminho_csv_2 = "polinjus.csv"

    # Lê o arquivo pulando as primeiras 1296 linhas e com separador ";"
    df2_polinjus_final = pd.read_csv(caminho_csv_2, skiprows=1296, sep=";")
    df2_polinjus_final = df2_polinjus_final[df2_polinjus_final['Usina'].notna()]
    df2_polinjus_final['QjusMax'] = pd.to_numeric(df2_polinjus_final['QjusMax'], errors='coerce')
    df2_polinjus_final_clean = df2_polinjus_final[['Usina', 'IndCurva', 'QjusMax', 'a0', 'a1', 'a2', 'a3', 'a4']]

    resultados = []
    for (usina_id, estagio), cortes_usina in df.groupby(["ID", "Estágio"]):
        usina_id_int = int(usina_id)
        uhe = Caso.hidr.get(usina_id_int)
        if uhe is None:
            continue
        q_max = 0
        p_max = 0
        v_vert = 0
        v_fixo =  uhe['vol_max']
        v_vert = uhe['vol_vert']
        h_loss = uhe['perda_hid']
        PCV =  uhe['pol_cota_vol']

        linha_usina = df2_polinjus_final_clean[df2_polinjus_final_clean['Usina'] == float(usina_id)]
        if not linha_usina.empty:
            polinomio_usina = linha_usina.iloc[-1]
            PVNJ = [polinomio_usina[f'a{i}'] for i in range(5)]
            d_max = polinomio_usina['QjusMax']
        else:
            PVNJ = uhe['pol_vaz_niv_jus']
            d_max = q_max

        for i in range(uhe['num_conj_maq']):
            q_max += uhe['maq_por_conj'][i] * uhe['vaz_efet_conj'][i]

        qued = HL(v_fixo, d_max, h_loss, PCV, PVNJ)
        qued_q = HL(v_fixo, q_max, h_loss, PCV, PVNJ)
        p_max = uhe['prod_esp'] * q_max * qued
        p_qmax = uhe['prod_esp'] * q_max * qued_q
        q_fixo = q_max
        p_fixo = p_max
        s_max = encontrar_limite_superior_S(q_fixo, v_fixo, p_fixo, cortes_usina)

        resultados.append({
            "ID da Usina": usina_id,
            "Estágio": estagio,
            "Nome da Usina": uhe['nome'],
            "Defluência Max Polinjus": d_max,
            "Potência Máxima N. Linear (S=0)": p_qmax,
            "Potência Máxima da N. Linear em Dmax": p_max,
            "Vertimento Maximo Smax (Secante)": s_max,
        })

    df_resultados = pd.DataFrame(resultados).sort_values(by=["ID da Usina", "Estágio"])
    df_resultados.to_excel("resultado_max_spillage.xlsx", index=False)
    print("Arquivo 'resultado_max_spillage.xlsx' gerado com sucesso.")

if __name__ == "__main__":
    'Carregando Dados:'
    Caso = Newave('NEWAVE')
    main()