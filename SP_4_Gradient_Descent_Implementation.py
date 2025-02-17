                                    # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.graph_objs as go
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.iolib.summary2 import summary_col
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder 
from statstests.process import stepwise

""" 1. Ingestão dos Dados """
database = pd.read_csv("transformed_data.csv")  # dataframe com as principais features
df = database.drop("Unnamed: 0",axis=1)

columns = df.columns


""" 2. Gradient Descent Implementation"""

def plot(index,tetas_0,tetas_1):
    # Configurações de estilo
    plt.rcParams['font.size'] = 12  # Tamanho da fonte
    plt.rcParams['axes.labelsize'] = 14  # Tamanho dos rótulos dos eixos
    plt.rcParams['axes.titlesize'] = 16  # Tamanho do título
    plt.rcParams['lines.linewidth'] = 2  # Espessura da linha
    plt.rcParams['lines.markersize'] = 6  # Tamanho dos marcadores

    # Criando a figura e os subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 linha, 2 colunas

    # Plotando o primeiro gráfico
    axs[0].plot(index, tetas_0, marker='o', markersize=4, color='blue', label=f'$\\theta_0 = {tetas_0[-1]:.2f}$')
    axs[0].set_xlabel('Iterações')
    axs[0].set_ylabel(r'$\theta_0$')
    axs[0].set_title('Variação de $\\theta_0$')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)  # Grade discreta

    # Plotando o segundo gráfico
    axs[1].plot(index, tetas_1, marker='o', markersize=4, color='green', label=f'$\\theta_1 = {tetas_1[-1]:.2f}$')
    axs[1].set_xlabel('Iterações')
    axs[1].set_ylabel(r'$\theta_1$')
    axs[1].set_title('Variação de $\\theta_1$')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)  # Grade discreta

    # Ajustando o layout
    plt.tight_layout()  # Evita sobreposição de elementos
    plt.show()




def gradient_descent(temp):
    w0,w1  = 0,0 # Pamameters os Lin. R
    lr = 0.1 # Learning Rate
    m = len(temp) # Number of Obs
    
    x_vals = temp['gmean_Valence'].values
    y_vals  = temp['critical_temp'].values
   
    tetas_0 = []
    tetas_1 = []
    index = []
    for j in range(200):
        dj_w0, dj_w1 = 0,0
        
        tetas_0.append(w0)  
        tetas_1.append(w1)    
        index.append(j)
        for i in range(0,m) : 
            x = x_vals[i] 
            y = y_vals[i]
            
            h = w0 + w1*x # Hypothesis Functions
            
            dj_w0 =+ (h - y)    # Derivative of Cost Function with Respect to W0
            dj_w1 =+ (h - y)*x  # Derivative of Cost Function with Respect to W1
 
        w0 =- lr*dj_w0
        w1 =- lr*dj_w1
        #print(f' W0 : {round(w0,2)} e W1 : {round(w1,2)} p/ iter. {j}')
    
    return tetas_0,tetas_1,index

def stochastic_gradient_descent(temp):
    w0,w1  = 0,0 # Pamameters os Lin. R
    lr = 0.02 # Learning Rate
    m = len(temp) # Number of Obs
    
    x_vals = temp['gmean_Valence'].values
    y_vals  = temp['critical_temp'].values
   
    tetas_0 = []
    tetas_1 = []
    index = []
    
    dj_w0, dj_w1 = 0,0
    
    tetas_0.append(w0)  
    tetas_1.append(w1)    
    
    for i in range(0,m) : 
        
        index.append(i)
        tetas_0.append(w0)  
        tetas_1.append(w1)  
        
        x = x_vals[i] 
        y = y_vals[i]
        
        h = w0 + w1*x # Hypothesis Functions
        
        dj_w0 = (h - y)    # Derivative of Cost Function with Respect to W0
        dj_w1 = (h - y)*x  # Derivative of Cost Function with Respect to W1

        
        w0 =- lr*dj_w0
        w1 =- lr*dj_w1
    index.append(m+1)
    #print(f' W0 : {round(w0,2)} e W1 : {round(w1,2)} p/ iter. {j}')
    
    return tetas_0,tetas_1,index


#tetas_0,tetas_1,index = gradient_descent(df[0:20000])

tetas_0,tetas_1,index = gradient_descent(df[0:20000])
plot(index,tetas_0,tetas_1)


tetas_0,tetas_1,index = stochastic_gradient_descent(df[0:20000])
plot(index,tetas_0,tetas_1)

