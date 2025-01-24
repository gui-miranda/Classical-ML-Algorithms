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

# Normalizando as variaveis métricas através do Z-Score (Médio 0 e Desvio P = 1)
# df_features = df[columns[1:-2]].sample(frac=1)

# for col in df_features.columns:
#     df_features.loc[:,col] = stats.zscore(df_features[col])

""" 2. GLM's """

""" 2.1 Regressão Linear - Simples """

# Utilizando a Feature - 'gmean_Valence'
temp = df[['gmean_Valence','critical_temp']]

plt.figure(figsize=(10,5))
sns.regplot(data=temp, x='gmean_Valence', y='critical_temp', ci=False)
plt.xlabel('gmean_Valence', fontsize=14)
plt.ylabel('Temperatura Crítica', fontsize=14)  
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=14)
plt.show()

#Estimação do modelo
modelo = sm.OLS.from_formula('critical_temp ~ gmean_Valence', temp).fit()

#Observação dos parâmetros resultantes da estimação
modelo.summary()

# Realizando uma transformação de Box-Cox 
x,lmbda = stats.boxcox(temp['critical_temp'])
temp['bc_critical_temp'] = x

plt.figure(figsize=(10,5))
sns.regplot(data=temp, x='gmean_Valence', y='bc_critical_temp', ci=False)
plt.xlabel('gmean_Valence', fontsize=14)
plt.ylabel('Temperatura Crítica', fontsize=14)  
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=14)
plt.show()

#Estimação do modelo
modelo = sm.OLS.from_formula('bc_critical_temp ~ gmean_Valence', temp).fit()

#Observação dos parâmetros resultantes da estimação
modelo.summary()

""" 2.1 Regressão Linear - Múltipla - Utilizando Step Wise"""
temp = df[columns[1:-1]]

string = 'critical_temp ~'
for c in temp.columns[0:-1] :
    string += f' {c} +'
string = string[0:-2]

modelo = sm.OLS.from_formula(string, temp).fit()

modelo_step = stepwise(modelo, pvalue_limit=0.05)

modelo_step.summary()

""" 2.1 Regressão Linear - Múltipla - Utilizando Box-Cox e Step Wise"""
temp = df[columns[1:-1]]

#Aplicando uma transformação de Box_Cox 
x,lmbda = stats.boxcox(temp['critical_temp'])
temp['bc_critical_temp'] = x

string = 'bc_critical_temp ~'
for c in temp.columns[0:-1] :
    string += f' {c} +'
string = string[0:-2]


modelo = sm.OLS.from_formula(string, temp).fit()

modelo_step = stepwise(modelo, pvalue_limit=0.05)

modelo_step.summary()

# Checando os resíduos 
temp['fitted'] =  (modelo_step.fittedvalues * lmbda + 1 ) ** (1 / lmbda) 
temp['erro']   = np.abs(temp['critical_temp'] - temp['fitted'])

from scipy.stats import norm
plt.figure(figsize=(10,8))
sns.histplot(temp['erro'], kde=True, bins=20,
             color='red')
plt.xlabel('Resíduos do Modelo Box-Cox', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


""" 2.1 Regressão Linear - Múltipla - Utilizando Box-Cox e Step Wise e Cross Validation """
from sklearn.linear_model import LinearRegression

from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score

temp = df[columns[1:-1]]

#Aplicando uma transformação de Box_Cox 
x,lmbda = stats.boxcox(temp['critical_temp'])
temp['bc_critical_temp'] = x

string = 'bc_critical_temp ~'
for c in temp.columns[0:-1] :
    string += f' {c} +'
string = string[0:-2]


lin_reg = LinearRegression()

scores = cross_val_score(lin_reg, temp[columns[1:-2]], temp['bc_critical_temp'],
 scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-scores)
lin_rmse_scores_adj = (lin_rmse_scores * lmbda + 1 ) ** (1 / lmbda) 

# Checando o comportamento do Erro Médio de acordo com o numero de Folds
means = []
n_folds = []
for i in range(2,11):
    lin_reg = LinearRegression()

    scores = cross_val_score(lin_reg, temp[columns[1:-2]], temp['bc_critical_temp'],
    scoring="neg_mean_squared_error", cv=i)

    lin_rmse_scores = np.sqrt(-scores)
    lin_rmse_scores_adj = (lin_rmse_scores * lmbda + 1 ) ** (1 / lmbda) 
    
    mean_rmse = np.mean(lin_rmse_scores_adj)
    
    n_folds.append(i)
    means.append(mean_rmse)

plt.figure(figsize=(10,8))
plt.plot(n_folds,means,marker='o')
plt.grid()
plt.xlabel('N Folds', fontsize=16)
plt.ylabel('Mean RMSE', fontsize=16)





# from sklearn import datasets, linear_model
# from sklearn.model_selection import cross_val_score
# diabetes = datasets.load_diabetes()
# X = diabetes.data[:150]
# y = diabetes.target[:150]
# lasso = linear_model.Lasso()
# print(cross_val_score(lasso, X, y, cv=3))

# X