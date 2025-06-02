"""
Estimate maximum spillage per hydro plant for which FPH >= 0
@author: dlbr
"""

import pandas as pd
import numpy as np
from PySDDP.Pen import Newave

def carregar_cortes(caminho_csv):
    """Lê os cortes da planilha CSV."""
    return pd.read_csv(caminho_csv, delimiter=';')

def fph(q, v, s, df):
    """Calcula o valor da função FPH (mínimo dos cortes) para (q, v, s)."""
    valores = df['Coef_Q'] * q + df['Coef_V'] * v + df['Coef_S'] * s
    return valores.min()

def encontrar_limite_superior_S(q, v, p_fixo, df, s_min=0, s_max=100000, tol=1e-3, max_iter=100):
    """
    Busca o maior valor de S tal que FPH(q, v, S) >= 0 usando busca binária.
    """
    baixo = s_min
    alto = s_max
    for _ in range(max_iter):
        meio = (baixo + alto) / 2
        potencia = fph(q, v, meio, df)
        if potencia >= p_fixo*(1-0.2): #20% de redução da potência
            baixo = meio
        else:
            alto = meio
        if abs(alto - baixo) < tol:
            break
    return baixo

def main():
    caminho_csv = 'fphs_cortes.csv'  # Substitua pelo caminho correto no seu PC
    df = carregar_cortes(caminho_csv)

    # Renomeia colunas
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={
        "ID da Usina": "ID",
        "Coeficiente de Volume": "Coef_V",
        "Coeficiente de Turbinamento": "Coef_Q",
        "Coeficiente de Vertimento": "Coef_S"
    })

    
    

    resultados = []
    for usina_id, cortes_usina in df.groupby("ID"):
        uhe = Caso.hidr.get(usina_id)
        q_max = 0
        p_max = 0
        v_vert = 0
        v_fixo =  uhe['vol_max']
        v_vert = uhe['vol_vert']
        s_disp = v_fixo - v_vert

        for i in range(uhe['num_conj_maq']):
            q_max=q_max+uhe['maq_por_conj'][i]*uhe['vaz_efet_conj'][i]
            p_max=p_max+uhe['maq_por_conj'][i]*uhe['pef_por_conj'][i]
        q_fixo = q_max
        p_fixo = p_max
        s_max = encontrar_limite_superior_S(q_fixo, v_fixo, p_fixo, cortes_usina)
        resultados.append({
            "ID da Usina": usina_id,
            "Nome da Usina": uhe['nome'],
            "Spillage Máximo com FPH >= 0.2xPmax": s_max,
            "Vol. Max (hm³)": uhe['vol_max'],
            "Vol. Vert (hm³)": uhe['vol_vert'],
            "Dif-Max-Vert (hm³)": s_disp
        })

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel("resultado_max_spillage.xlsx", index=False)
    print("Arquivo 'resultado_max_spillage.xlsx' gerado com sucesso.")

if __name__ == "__main__":
    'Carregando Dados:'
    Caso = Newave('NEWAVE') #Lê os dados do Newave    
    main()



