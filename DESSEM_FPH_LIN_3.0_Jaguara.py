# -*- coding: utf-8 -*-awzsdxrcfgtvnuhyjolç

# Author: David 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import math
from random import seed
import random
from PySDDP.Pen import Newave
#import plotly.graph_objects as go
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error as mape
#from sklearn.metrics import maximum_absol
from sklearn.metrics import mean_squared_error as mse
#import plotly.graph_objects as go
#import plotly.offline as pyo
#from tensorflow import keras
from rdp import rdp
import time
from scipy.spatial import ConvexHull
from gurobipy import *
from sympy.combinatorics.graycode import GrayCode



#%%
class DataPrep():
    
    def __init__(self):
        return 0
        
class FPH():
    
    def __init__(self):
        np.random.seed(0)               
        'Coeficientes do Polinômio'
        self.PCA = uhe['pol_cota_area'] #Cota Áreas
        self.PCV = uhe['pol_cota_vol'] #Cota volume
        
        #self.PCV = [620.4570922851562,  1e-40, 0.0, 0.0, 0.0]
        #self.PCV = [885.6586303710938, 1e-40, 0.0, 0.0, 0.0] #camargos
        #self.PCV = [807.9130859375, 1e-40, 0.0, 0.0, 0.0] #FUNIL-GRANDE
        self.PVNJ = uhe['pol_vaz_niv_jus'] #Cota volumeuhe['pol_vaz_niv_jus']
        #self.PVNJ = [261.363,	0.00301186,	-5.636080E-7,	6.791440E-11,	-3.028480E-15] #Cota volumeuhe['pol_vaz_niv_jus']
        
    
    def FCM(self, Volume):
        cota_montante = 0
        for i in range(len(self.PCV)):
            cota_montante += self.PCV[i]*Volume**i               
        return cota_montante
    
    def FCJ(self, Defluencia):
        cota_jusante = 0
        for i in range(len(self.PVNJ)-1):
            cota_jusante += self.PVNJ[i]*Defluencia**i               
        return cota_jusante

    def HL(self, Volume, Defluencia, w):
        return self.FCM(Volume)-self.FCJ(Defluencia)-uhe['perda_hid']
   
    def RDP(self, fph):
                   
        fph = rdp(fph,epsilon=0.2) 
       
        return fph   
    def fph_out(self, disc, Estratégia = 'Agregada', rdp = False, *args, **kwargs):
        self.Estratégia = Estratégia       
        self.fph = list()
        vert = kwargs.get('vert', False)
        if Estratégia == 'Agregada':
            if not(kwargs.get('NUG', None)):
                #UG = uhe['maq_por_conj'][0]
                NUG =  sum(uhe['maq_por_conj'])
            else:
                NUG = kwargs.get('NUG', None)
        if Estratégia == 'Individual':
            NUG=1
       
        self.q_max = 0
        for i in range(uhe['num_conj_maq']):
            self.q_max=self.q_max+uhe['maq_por_conj'][i]*uhe['vaz_efet_conj'][i]
            
        #Discretização
        self.vazao_usina = np.linspace(0, self.q_max, disc[0])
        #self.vol_var = np.linspace(uhe['vol_min'], uhe['vol_max'], disc[1]) #
        
        self.vol_var = np.linspace(Vini*(1-0.00001),Vini*(1+0.00001))
        if Reg == 'M':
            self.vol_var = np.linspace(max(uhe['vol_min'], Vini - (1/10)*(uhe['vol_max']-uhe['vol_min'])), min(uhe['vol_max'], Vini + (1/10)*(uhe['vol_max']-uhe['vol_min'])), disc[1])             
        if Reg=='SR':
            #self.vol_var = np.linspace(max(uhe['vol_min'], Vini - (1/100)*(uhe['vol_max']-uhe['vol_min'])), min(uhe['vol_max'], Vini + (1/100)*(uhe['vol_max']-uhe['vol_min'])), disc[1]) 
            self.vol_var = np.linspace(Vini - (1/100)*(uhe['vol_min']),  Vini + (1/100)*(uhe['vol_max']), disc[1])         
        if Reg=='S':
            self.vol_var = np.linspace(max(uhe['vol_min'], Vini - (2/10)*(uhe['vol_max']-uhe['vol_min'])), min(uhe['vol_max'], Vini + (2/10)*(uhe['vol_max']-uhe['vol_min'])), disc[1])        
        
                
            
        #FPH Não Linear     
        for vaz in self.vazao_usina:
            pot = 0
           # vaz = vaz + uhe['vaz_efet_conj'][0]*(NUG-1)
            for vol in self.vol_var:                
                for i in range(NUG):
                        w = vaz/(i+1)
                        qued = self.HL(vol, vaz, w)                       
                        pot_n = uhe['prod_esp']*w*qued
                        por_n_max = (i+1)*pot_n
                        pot = np.max([pot, por_n_max])
                self.fph.append([vaz, vol, pot])
        if rdp:
            self.fph=self.RDP(self.fph)
                   
        return self.fph    
   
    def fph_out_s(self, disc, Estratégia = 'Agregada', rdp = False, vqmax=True, *args, **kwargs):
        self.Estratégia = Estratégia       
        self.fph = list()
        vert = kwargs.get('vert', False)

        if Estratégia == 'Agregada':
            if not(kwargs.get('NUG', None)):
               #NUG = uhe['maq_por_conj'][0]
               NUG =  sum(uhe['maq_por_conj'])
            else:
                NUG = kwargs.get('NUG', None)
        if Estratégia == 'Individual':
            NUG=1
        
         #FPH com vertimento                

        self.vazao_usina = np.linspace(0, self.q_max, disc[0])
        self.vol_var = np.linspace(uhe['vol_min'], uhe['vol_max'], disc[1]) #
        #self.vol_var = np.linspace(uhe['vol_min']*(1-1e-40), uhe['vol_max']*(1+1e-40), disc[1]) #
       
        #self.vol_var = np.linspace(Vini*(1-0.00001),Vini*(1+0.00001))
        if Reg == 'M':
            self.vol_var = np.linspace(max(uhe['vol_min'], Vini - (1/10)*(uhe['vol_max']-uhe['vol_min'])), min(uhe['vol_max'], Vini + (1/10)*(uhe['vol_max']-uhe['vol_min'])), disc[1])             
        if Reg=='SR':           
            self.vol_var = np.linspace(Vini - (1/100)*(uhe['vol_min']),  Vini + (1/100)*(uhe['vol_max']), disc[1])         
        if Reg=='S':
            self.vol_var = np.linspace(max(uhe['vol_min'], Vini - (2/10)*(uhe['vol_max']-uhe['vol_min'])), min(uhe['vol_max'], Vini + (2/10)*(uhe['vol_max']-uhe['vol_min'])), disc[1])        
        
           
        self.vert_var = np.linspace(0, vaz_ext-max(self.vazao_usina), disc[2])           
        if vqmax == True:
            self.vazao_usina = np.linspace(max(self.vazao_usina), max(self.vazao_usina), 1) 
            self.vol_var = np.linspace(max(self.vol_var), max(self.vol_var), 1)
            
           # self.vert_var = np.linspace(0, self.q_max*2, disc[2]) 
                       
        for vaz in self.vazao_usina:            
            for vol in self.vol_var:
                for vert in self.vert_var:
                    pot = 0                     
                    for i in range(NUG):
                            w = vaz/(i+1)
                            defluencia = vaz + vert
                            qued = self.HL(vol, defluencia , w)
                            pot_n = uhe['prod_esp']*w*qued
                            por_n_max = (i+1)*pot_n
                            pot = np.max([pot, por_n_max])
                    self.fph.append([vaz, vol, vert, pot])
                  
        return self.fph    
    
    def Plota_FPH(self, fph):
       fph = np.array(fph)        
       x = fph[:,0]
       y = fph[:,1]
       z = fph[:,2]

       fig2 = plt.figure()
       ax = fig2.add_subplot(1,1,1, projection='3d')
       #ax = fig2.gca(projection='3d')
       plt.title('Função de Produção Hidrelétrica da Usina ' + uhe['nome'])
       ax.yaxis.set_tick_params(labelsize=10)
       ax.set_xlabel('Vazão Turbinada (m³/s)')
       ax.set_ylabel('Volume (hm³)')
       ax.set_zlabel('Potência Gerada (MW)')
       ax.dist = 11

       ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
       plt.tight_layout()
           
 
       
    def Plota_FCM(self):
        fcm = list()
        for vol in self.vol_var:
             fcm.append(self.FCM(vol))
        
        fig3 = plt.figure()  
        plt.plot(self.vol_var , fcm,'-')
        plt.legend(loc=2, fontsize='small')
        plt.title('Polinomio Cota-Volume da Usina ' + uhe['nome'])
        plt.xlabel("Volume do Reservatório (hm³)")
        plt.ylabel("Cota  (metros)")
        plt.grid()
        plt.show()
        
    def Plota_FCJ(self):
        fcj= list()
        for vaz in self.vazao_usina:
             fcj.append(self.FCJ(vaz))
        
        fig4 = plt.figure()  
        plt.plot(self.vazao_usina , fcj,'-')
        plt.legend(loc=2, fontsize='small')
        plt.title('Polinomio Vazão-Nível Jusante da Usina ' + uhe['nome'])
        plt.xlabel("Defluência (m³/s)")
        plt.ylabel("Cota  (metros)")
        plt.grid()
        plt.show()

#%%
class FPH_Linear(): 
    
    def __init__(self):
        self.fph = list()
        self.q_max = 0
        for i in range(uhe['num_conj_maq']):
            self.q_max=self.q_max+uhe['maq_por_conj'][i]*uhe['vaz_efet_conj'][i]
    
    def PWL_CHULL(self, disc, Estratégia, rdp=False, *args, **kwargs):
        
        if Estratégia == 'Agregada':
            if not(kwargs.get('NUG', None)):
                #self.NUG = uhe['maq_por_conj'][0]
                self.NUG = sum(uhe['maq_por_conj'])
            else:
                self.NUG = kwargs.get('NUG', None)
        if Estratégia == 'Individual':
            self.NUG = 1
            
        self.fph = FPH.fph_out(disc, Estratégia, rdp=rdp, NUG = self.NUG)
        fph = self.fph
        self.fph_s = FPH.fph_out_s(disc, Estratégia, rdp=rdp, vqmax = True)
        fph_s = self.fph_s
        M = len(fph_s)

        N = len(fph)
        Pontos =[]
        #coef_ajust = 1
        Pontos.append(self.fph)
        
        Planos = []
        NTugs = 1 #Futuramente sera alterado para compreender usinas com mais de dois grupos de unidades
        for i in range(NTugs):
            Planos.append(ConvexHull(Pontos[i]).equations) # as grandezas saem na mesma sequência que a do vetor "Pontos", nesse caso: (HB, W, PG , termo independente)

        ## Excluindo Equações que dizem que "PG*0"
        excluir1 = []
        for i in range(NTugs):
            excluir1_aux = []
            for j in range(len(Planos[i])):
                if Planos[i][j][2] != 0:
                    Planos[i][j] = -Planos[i][j]/Planos[i][j][2]
                else:
                    excluir1_aux.append(j)
            excluir1.append(np.array(excluir1_aux, dtype = np.int64))
            ## Deletando linhas (equações fornecidas pelo CH)
            Planos[i] = np.delete(Planos[i], excluir1[i], axis = 0)
            
        excluir = []
        Dif = []
        for i in range(NTugs):
            Wint = []
            Wint_aux = []
            HBint = []
            HBint_aux = []
            PGaux = []
            excluir_aux = []
            #Pontos_aux = []
            for p in range(len(Planos[i])):
                if p == 0:
                    for k in range(len(Pontos[i])):
                        Wint_aux.append(Pontos[i][k][1])
                        HBint_aux.append(Pontos[i][k][0])
                Wint.append(np.array(Wint_aux, dtype = np.float64)) #np.arange(Pontos[0][0][0], Pontos[i][len(Pontos[0]) - 1][0] + 0.0001, round(Pontos[0][1][0] - Pontos[0][0][0],8)))
                HBint.append(np.array(HBint_aux, dtype = np.float64))
                # Potência do hull em todos os pontos "qs" fornecidos
                PGaux.append(np.array(Planos[i][p][0]*HBint[p] + Planos[i][p][1]*Wint[p] + Planos[i][p][3], dtype = np.float64))
                #Pontos_aux.append(np.array([HBint[p], Wint[p]]))
                
            # Comparando o "p" do hull com o "p" original
            for q in range(len(Planos[i])):
                for r in range(len(Wint[i])):
                    if round(PGaux[q][r],7) < round(Pontos[i][r][len(Pontos[i][r])-1],7):
                        excluir_aux.append(q)
                        break
            ## Obtendo a diferença entre o p original e o p aproximado
            for q in range(len(Planos[i])):
                Difaux = []
                for r in range(len(Wint[i])):
                    Difaux.append(PGaux[q][r] - Pontos[i][r][len(Pontos[i][r])-1])
                if i == 0:
                    Dif.append(np.array(Difaux, dtype = np.float64))
        # Adicionando os planos a serem excluidos
            excluir.append(np.array(excluir_aux, dtype=np.int64))
        ## Deletando linhas (equações fornecidas pelo CH)
            Planos[i] = np.delete(Planos[i],excluir[i], axis = 0)            
          #  Pontos_aux[i] = np.delete(Pontos_aux[i],excluir[i], axis = 0) 
        self.coef = Planos
        self.coef = np.delete(self.coef, 2, 2)[0]
   
        n= Model("vert")
        k = Model("ajuste")
        fobj, Y = [], []
        y = n.addVar(lb=-1000, ub=1000, name="S_coef")
        coef_a = k.addVar(lb=0, ub=1, name="coef_ajuste")

    #----------------------------------------------------------
    #-------------------- RESTRIÇÕES --------------------------
    #----------------------------------------------------------
        coeficientes = []
        ajuste = 1
    # Objetive Function 
    #Correção FPHA

        if FPHA_Adj:
            k.setObjective((1/N)*(quicksum(fph[i][-1]-coef_a*(self.lin_fph(fph[i][0],fph[i][1])) for i in range(N))**2), GRB.MINIMIZE)  
            k.write("retricoes_corretor.lp")
            k.Params.timeLimit = 120
            k.params.MIPGap = 0.0000001
            k.optimize()   
                             
            if k.status == GRB.Status.OPTIMAL:
                fobj = k.objVal
                ajuste = coef_a.X
            #coeficientes.append([co[0], co[1], co[2]])    #Desativar caso ative o código acima
        self.coef = self.coef*ajuste
        #Adição Vertimento      
        for co in self.coef:
            n.setObjective((1/M)*(quicksum(fph_s[i][-1]-(self.lin_fph_s(fph_s[i][0],fph_s[i][1],co)+y*fph_s[i][2]) for i in range(M))**2), GRB.MINIMIZE)  
            n.write("retricoes_vertimento_sec.lp")
            n.Params.timeLimit = 120
            n.params.MIPGap = 0.0000001
            n.optimize()   
                             
            if n.status == GRB.Status.OPTIMAL:
                fobj = n.objVal
                CB = y.X
            coeficientes.append([co[0], co[1], CB, co[2]])    #Desativar caso ative o código acima
                      
        self.coef_s = np.array(coeficientes)
        fph, fphl, acc  = self.fph_out_linear(Estratégia) #somente para q, v, s
        fph, fphl, acc_s  = self.fph_out_linear_s(Estratégia) #somente para q, v, s
        #fph, fphl, acc  = self.fph_out_linear(Estratégia) #somente para q, v, s
        acc_s = acc_s + acc
        
        return self.coef, self.coef_s, acc, acc_s, fph, fphl, fph_s, ajuste
        
    
    def lin_fph_s(self, q, v, co):
        coef = co
        return q*coef[0]+v*coef[1]+coef[2]
   
    def lin_fph(self, q, v):
        coef = self.coef
        plano_l = []
        for k in range(len(coef)):
                plano_l.append(np.matmul(coef[k,:],[q, v, 1]))
                fph_l=min(plano_l)
        return fph_l
       
    def fph_out_linear(self, Estratégia):
        fph = np.array(FPH.fph_out([100,100], Estratégia, rdp=False, NUG = self.NUG )) 
        fphl = np.array(FPH.fph_out([100,100], Estratégia, rdp=False, NUG = self.NUG ))
        #fph = np.array(FPH.fph_out(disc, Estratégia, rdp=False, NUG = self.NUG )) 
        #fphl = np.array(FPH.fph_out(disc, Estratégia, rdp=False, NUG = self.NUG ))
        #fphl = fph
        coef = self.coef
        if len(coef)>=0:
            for i in range(len(fphl)):
                q = fphl[i,0]
                v = fphl[i,1]
                plano_2 = []
                for k in range(len(coef)):
                    plano_2.append(np.matmul(coef[k,:],[q, v, 1]))
                    fphl[i,2]=min(plano_2)
        #acc = r2_score(fph[:,-1], fphl[:,-1])
        #acc = mean_absolute_error(fph[:,-1], fphl[:,-1])
        #acc = mape(fph[:,-1], fphl[:,-1])
        acc = self.mapes(fph[:,-1], fphl[:,-1])
        #acc = mean_absolute_error(fph[:,-1], fphl[:,-1])
        amostra = len(fphl)
        #print(f"Usina {uhe['nome'].strip()}: A Precisão da Linearização é de {acc*100}%")  
        print(f"Usina {uhe['nome'].strip()}: O Erro Médio Absoluto é de {acc}")
        #print(f"Usina {uhe['nome'].strip()}: A Precisão da Linearização é de {acc}")                      
        return fph, fphl, acc     
    
    def mapes(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)    
    # Ignorar divisões por zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def fph_out_linear_s(self, Estratégia):
        fph = np.array(FPH.fph_out_s([100, 100, 100], Estratégia, rdp=False,vqmax =True,  NUG = self.NUG )) 
        fphl = np.array(FPH.fph_out_s([100, 100, 100], Estratégia, rdp=False, vqmax =True,  NUG = self.NUG )) 
        #fph = np.array(FPH.fph_out_s(disc, Estratégia, rdp=False,vqmax =False,  NUG = self.NUG )) 
        #fphl = np.array(FPH.fph_out_s(disc, Estratégia, rdp=False, vqmax =False,  NUG = self.NUG ))
        #fphl = fph
        coef = self.coef_s
        if len(coef)>=0:
            for i in range(len(fphl)):
                q = fphl[i,0]
                v = fphl[i,1]
                s = fphl[i,2]
                plano_2 = []
                for k in range(len(coef)):
                    plano_2.append(np.matmul(coef[k,:],[q, v,  s, 1]))
                    # if q >= self.q_max and v >= uhe['vol_max']:                        
                    #     plano_2.append(np.matmul(coef[k,:],[q, v,  s, 1]))
                    # else:
                    #     plano_2.append(np.matmul(coef[k,:],[q, v,  0, 1]))
                    fphl[i,3]=min(plano_2)
        #acc = r2_score(fph[:,-1], fphl[:,-1])
        #acc = mean_absolute_error(fph[:,-1], fphl[:,-1])
        acc = self.mapes(fph[:,-1], fphl[:,-1])
        #acc = mape(fph[:,-1], fphl[:,-1])
        #acc = mse(fph[:,-1], fphl[:,-1])
        amostra = len(fphl)
        print(f"Usina {uhe['nome'].strip()}: A Erro Médio Absoluto (com vertimento) é de {acc}") 
       # print(f"Usina {uhe['nome'].strip()}: A Precisão da Linearização é de {acc*100}%")   
                     
        return fph, fphl, acc  
#%%
class FPH_SEL():
    def __init__(self):
        self.coef_fph=list()   
        
    def MOD1(self, disc, PWL = 'CH', rdp = False, Mostraracc=False, *args, **kwargs):
        
        self.coef_fph=list()
        acc_fphl = list()       
        PH = kwargs.get('PH', None)
        NUGM = kwargs.get('NUGM', None)
        
        if PH == None:
            PH = 3
            
        if NUGM == None:          
            for NG in range(uhe['maq_por_conj'][0]):
                if PWL == 'CH':
                    coef, coef_s, acc, fph, fphl, fph_s = FPH_Linear.PWL_CHULL( disc, Estratégia, rdp=rdp, NUG = NG)
                    if Mostraracc:
                        #fph_n_linear, fph_linear, acc = FPH_Linear.fph_out_linear(Estratégia)
                        fph_n_linear, fph_linear, acc = FPH_Linear.fph_out_linear_s(Estratégia)
                        acc_fphl.append(acc)
                if PWL == 'MQ':
                    #c = FPH_Linear.PWL_MQ(disc, Estratégia, PH, rdp=rdp, NUG = NG)
                    coef, coef_s, acc, fph, fphl, fph_s = FPH_Linear.PWL_CHULL( disc, Estratégia, rdp=rdp, NUG = NG)
                    if Mostraracc:
                        #fph_n_linear, fph_linear, acc = FPH_Linear.fph_out_linear(Estratégia)
                        fph_n_linear, fph_linear, acc = FPH_Linear.fph_out_linear_s(Estratégia)
                        acc_fphl.append(acc)
                self.coef_fph.append(coef_s)
        else:            
                if PWL == 'CH':
                    coef, coef_s, acc, fph, fphl, fph_s = FPH_Linear.PWL_CHULL( disc, Estratégia, rdp=rdp, NUG = NUGM)
                    if Mostraracc:
                        fph_n_linear, fph_linear, acc = FPH_Linear.fph_out_linear(Estratégia)
                        # acc_fphl.append(acc)
                if PWL == 'MQ':
                    # c = FPH_Linear.PWL_MQ(disc, Estratégia, PH, rdp=rdp, NUG = NUGM)
                    coef, coef_s, acc, fph, fphl, fph_s = FPH_Linear.PWL_CHULL( disc, Estratégia, rdp=rdp, NUG = NUGM)
                    if Mostraracc:
                        fph_n_linear, fph_linear, acc = FPH_Linear.fph_out_linear(Estratégia)
                        acc_fphl.append(acc)
                self.coef_fph.append(coef_s)
                
        return  self.coef_fph, acc_fphl
     
#%%
#if __name__ == '__main__':
#--------------------------------------------------Carregada Dados----------------------------------------------------------------------------------------------------------------#
'Carregando Dados:'
Caso = Newave('NEWAVE') #Lê os dados do Newave
   
#uhe = Caso.hidr.get('ITA')        
#uhe = Caso.confhd.get('CAMARGOS')         
#uhe = Caso.hidr.get('ITUTINGA') 
#uhe = Caso.hidr.get('FUNIL-GRANDE') 
#uhe = Caso.hidr.get('FURNAS')
#uhe = Caso.confhd.get('M. DE MORAES')
#uhe = Caso.confhd.get('ESTREITO')
uhe = Caso.confhd.get('JAGUARA')
#Volume

Vini = uhe['vol_min'] + (1/2)*(uhe['vol_max']-uhe['vol_min']) #Cenário 1
#aaa
#%%
#--------------------------------------------------Incializa Modelo e Regressor do Polinômio de Rendimento Hidráulico----------------------------------------------------------------------------------------------------------------# 

FPH = FPH()  
'Estratégia: Individual ou Agregada:'
#uhe['num_pol_vnj']
#Estratégia = 'Individual'
Estratégia = 'Agregada'
Reg = 0
#Reg = 'SR'
#Reg  = uhe['tipo_reg']
FPHA_Adj = True
vaz_ext = 14500
#grp = 2


#Extração de pontos FPH
fph = FPH.fph_out([5,5], Estratégia = 'Agregada', rdp=False)
#fph = FPH.fph_out([10,10], Estratégia = 'Agregada', rdp=False, NUG = 2)
#FPH.Plota_Rendimento()
FPH.Plota_FPH(fph)
FPH.Plota_FCM()
FPH.Plota_FCJ()
#aaa
#%%
#--------------------------------------------------Inicializa Parâmetros de Linearização----------------------------------------------------------------------------------------------------------------# 
FPH_Linear = FPH_Linear()
#coef  =  FPH_Linear.PWL_CHULL([5,5,2], Estratégia, rdp=False)
#coef  =  FPH_Linear.PWL_CHULL([5,5,2], Estratégia, rdp=False, NUG = 1)

coef, coef_s, acc, acc_s, fph, fph1, fph_s, co  =  FPH_Linear.PWL_CHULL([5,5,2], Estratégia, rdp=False)

len(coef_s)


#coef, coef_s, acc,  fph, fph1, fph_s, co  =  FPH_Linear.PWL_CHULL([5,5,2], Estratégia, rdp=False)
#len(coef)
#coef = FPH_Linear.PWL_MQ([5,5], Estratégia, 4, rdp=False, NUG = 1)




fph_n_linear, fph_linear, acc = FPH_Linear.fph_out_linear(Estratégia)
FPH.Plota_FPH(fph_n_linear)
FPH.Plota_FPH(fph_linear)

#%%
# #DIsc x Erro
# # #disc_q = [2, 4, 5, 6, 8, 10, 20, 30, 40, 50]
# disc_v = [1]

# disc_q = [3, 4, 5, 6, 8, 10, 20]
# #disc_v = [2, 4, 5, 6, 8, 10, 20]

# ar=np.zeros([len(disc_v), len(disc_q)])
# ac=np.zeros([len(disc_q), len(disc_q)])
# ac_s=np.zeros([len(disc_v), len(disc_q)])

# for i in range(len(disc_q)):
#     for j in range (len(disc_v)):
#         coef, coef_s, acc, acc_s, fph, fph1, fph_s, co  =  FPH_Linear.PWL_CHULL([disc_q[i],disc_v[j],2], Estratégia, rdp=False)
#         ar[j, i]=acc
#         ac_s[j, i]=acc_s
#         ac[j, i]=len(coef)

# # Convert the arrays to DataFrames
# df1 = pd.DataFrame(ar)
# df2 = pd.DataFrame(ac)
# df3 = pd.DataFrame(ac_s)
# # Create a Pandas Excel writer object
# with pd.ExcelWriter('output.xlsx') as writer:
#     df1.to_excel(writer, sheet_name='Array1', index=False)
#     df2.to_excel(writer, sheet_name='Array2', index=False)
#     df3.to_excel(writer, sheet_name='Array3', index=False)
# print("Arrays have been written to output.xlsx")


############################################Escrever Resultados da Linearização##########################################################################################################    

#Q,V,S###############################################################################################################

Corte =  np.linspace(0, len(coef_s), len(coef_s)+1)

# Q1 = np.array(fph[:,0])
# V1=  np.array(fph[:,1])
# S1 = np.array(fph[:,2])
# P1 = np.array(fph[:,3])
# P2 = np.array(fph1[:,3])

# columns_2=['Q','V','S','PG','PGLinear']
# df2 = pd.DataFrame(list(zip(Q1, V1, S1, P1, P2)), columns=columns_2)

   
# with pd.ExcelWriter('FPH-LinearxNLinear.xlsx') as writer:
#       df2.to_excel(writer, sheet_name='FPHxFPHL'+uhe['nome'])

Q1 = np.array(fph[:,0])
V1=  np.array(fph[:,1])
#S1 = np.array(fph[:,2])
P1 = np.array(fph[:,2])
P2 = np.array(fph1[:,2])

columns_2=['Q','V','PG','PGLinear']
df2 = pd.DataFrame(list(zip(Q1, V1, P1, P2)), columns=columns_2)

   
with pd.ExcelWriter('FPH-LinearxNLinear.xlsx') as writer:
      df2.to_excel(writer, sheet_name='FPHxFPHL'+uhe['nome'])


Corte = np.linspace(0, len(coef_s), len(coef_s) + 1)
Q = np.array(coef_s[:, 0])
V = np.array(coef_s[:, 1])
S = np.array(coef_s[:, 2])
I = np.array(coef_s[:, 3])


TSF = np.empty(len(coef_s))*np.nan
# TSF1 = np.empty(len(coef_s))*np.nan
# TSF2 = np.empty(len(coef_s))*np.nan
# TSF[0] = acc
# TSF1[0] = max(fph_n_linear[:,2])
# #Perda=((uhe['perda_hid']*2.6)/(uhe['vaz_efet_conj'][0]**2))
# print(TSF1[0])
# #print(Perda)

# Use square brackets to create a list with a single value for 'R²'
columns_3 = ['Corte', 'Coef_Q', 'Coef_V', 'Coef_S', 'Coef_Independente', 'MAPE']
df3 = pd.DataFrame(list(zip(Corte, Q, V, S, I, TSF)), columns=columns_3)
print(df3)
with pd.ExcelWriter('FPH-Coef-'+uhe['nome']+'-.xlsx') as writer:
    df3.to_excel(writer, sheet_name='Coef_FPH')
#Q,V###############################################################################################################

# Corte =  np.linspace(0, len(coef), len(coef)+1)

# Q1 = np.array(fph[:,0])
# V1=  np.array(fph[:,1])
# P1 = np.array(fph[:,2])
# P2 = np.array(fph1[:,2])

# columns_2=['Q','V','PG','PGLinear']
# df2 = pd.DataFrame(list(zip(Q1, V1, P1, P2)), columns=columns_2)

   
# with pd.ExcelWriter('FPH-LinearxNLinear.xlsx') as writer:
#       df2.to_excel(writer, sheet_name='FPHxFPHL'+uhe['nome'])

# Corte = np.linspace(0, len(coef), len(coef) + 1)
# Q = np.array(coef[:, 0])
# V = np.array(coef[:, 1])
# I = np.array(coef[:, 2])


# TSF = np.empty(len(coef))*np.nan
# TSF1 = np.empty(len(coef))*np.nan
# TSF2 = np.empty(len(coef))*np.nan
# TSF[0] = acc
# TSF1[0] = max(fph_n_linear[:,2])

# columns_3 = ['Corte', 'Coef_Q', 'Coef_V', 'Coef_Independente', 'MAPE']
# df3 = pd.DataFrame(list(zip(Corte, Q, V, I, TSF)), columns=columns_3)
# print(df3)
# # Use square brackets to create a list with a single value for 'R²'
# with pd.ExcelWriter('FPH-Coef-'+uhe['nome']+'-.xlsx') as writer:
#     df3.to_excel(writer, sheet_name='Coef_FPH')
    
#Q,HB###############################################################################################################
#HB
# Corte =  np.linspace(0, len(coef), len(coef)+1)
   
# with pd.ExcelWriter('FPH-LinearxNLinear.xlsx') as writer:
#       df2.to_excel(writer, sheet_name='FPHxFPHL'+uhe['nome'])

# Corte = np.linspace(0, len(coef), len(coef) + 1)
# Q = np.array(coef_s[:, 0])
# HB = np.array(coef_s[:, 1])
# I = np.array(coef_s[:, 2])


# TSF = np.empty(len(coef))*np.nan
# TSF1 = np.empty(len(coef))*np.nan
# TSF2 = np.empty(len(coef))*np.nan
# TSF[0] = acc
# TSF1[0] = max(fph_n_linear[:,2])
# Perda=((uhe['perda_hid']*2.6)/(uhe['vaz_efet_conj'][0]**2))
# print(TSF1[0])
# print(Perda)

# # Use square brackets to create a list with a single value for 'R²'

# columns_3 = ['Corte', 'Coef_Q', 'HB', 'Coef_Independente', 'R²']
# df3 = pd.DataFrame(list(zip(Corte, Q, HB, I, TSF)), columns=columns_3)
# print(df3)

# with pd.ExcelWriter('FPH-Linear-Agregada'+'-'+uhe['nome']+'.xlsx') as writer:
#     df3.to_excel(writer, sheet_name=uhe['nome']+'-FPH(q, hb)')

'Volume Mínimo'
#print(f"Nome {uhe['nome'].strip()} - Volume Mínimo: {uhe['vol_min']}")

'Keys'
#print(uhe.keys())

'Histórico Vazão'
#Caso.confhd.plot_vaz(uhe)

'Código da Usina'
#print(f"A usina {uhe['nome'].strip()} possui o código {uhe['codigo']}") 

'Usina  Jusante'
#print(f"O código da usina à jusante de {uhe['nome'].strip()} é {uhe['jusante']}")
#uhe_jusante = Caso.confhd.get(uhe['jusante'])
#print(uhe_jusante['nome'])


#Caso.confhd.plot_pcv(uhe)

#Caso.confhd.plot_pca(uhe)
'Variação Volume Meses de Estudo'

#Caso.confhd.plota_volume(uhe)
# Defluencia = 3000
# cota_jusante_1 = 0
# cota_jusante_2 = 0
# for i in range(len(uhe['pol_vaz_niv_jus'])):
#     cota_jusante_1 = 261.36279296875
#     cota_jusante_2 += uhe['pol_vaz_niv_jus'][i]*Defluencia**i 

# cota_montante_1 = 0
# cota_montante_2 = 0
# PCV = [331.649, 0.00752020, 0.0, 0.0, 0.0] 
# for i in range(len(PCV)):
#     cota_montante_1 += PCV[i]*4300**i
#     cota_montante_2 += PCV[i]*5100**i

# print(cota_montante_2-cota_jusante_1 )
# print(cota_montante_1-cota_jusante_2 )

# uhe['vaz_efet_conj'][0]
# uhe['num_conj_maq']
# uhe['maq_por_conj']



# uhe['vaz_efet_conj']
# uhe['maq_por_conj']
# uhe['vol_min']
# uhe['vol_max']
