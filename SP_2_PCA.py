# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg

""" 1. Ingestão dos Dados """
database = pd.read_csv("transformed_data.csv")  # dataframe com as principais features
columns = database.columns
df = database
""" 2. Algoritmos Não Supervisionados """

""" 2.1 Redução de Dimensionalidade - PCA """

# Testando a adequação e possibilidade de extração de componentes principais
# nesse conjunto de dados, através do Teste de Bartllet.
# O Teste de Bartllet avalia a hipótese de que as amostras do conjunto possuem variância iguais.
# H0 : Todas as Variâncias são iguais na amostra (Não há FATORES). 
# H1 : Pelo menos 2 das amostras possuem variância distinta (Há FATORES)
# Se p_value < 5%, REJEITA-SE H0

df_features = df[columns[0:-2]]
bartlett, p_value = calculate_bartlett_sphericity(df_features)

print(f'Bartlett statistic: {bartlett}')

print(f'p-value : {p_value}')

# Checando os número de Fatores construidos e seus AUTOVALORES
fa = FactorAnalyzer()
fa.fit(df_features)

ev, v = fa.get_eigenvalues()
print(ev)

# Pelo critério da Raiz Latente, toma-se apenos fatores com 
# AUTOVALORES >= 1 (Nesse caso, apenas os 3 primeiros)

fa.set_params(n_factors = 3, method = 'principal', rotation = None)
fa.fit(df_features)

# Calculando Autovalores, Variancias, e Variancias Acm.
eigen_fatores = fa.get_factor_variance() 

tabela_eigen = pd.DataFrame(eigen_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T
tabela_eigen

# Calculando os Fatores para os observações do conjunto
predict_fatores= pd.DataFrame(fa.transform(df_features))
predict_fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(predict_fatores.columns)]
predict_fatores

df = pd.concat([df.reset_index(drop=True), predict_fatores], axis=1) #Adicionando no dataset

# Mensurando o Score Fatorial de cada váriavel dentro dos respectivos fatores
# através da memoria de calculo do predict

scores = fa.weights_
tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = df_features.columns
tabela_scores



# Calculando as Cargas Fatoriais
cargas_fatores = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatores)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = df_features.columns
tabela_cargas



# Calculando as Comunalidades (% da Variancia total da feature que foi capturada nos fatores utilizados)
comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = df_features.columns
tabela_comunalidades.sort_values(by='Comunalidades',ascending = True)


# Checando a Correlação Entre Fatores
corr_fator = pg.rcorr(df[['Fator 1','Fator 2','Fator 3']], method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})
print(corr_fator)



# Construindo o plot das Cargas Fatoriais para cada váriável 
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

tabela_cargas_chart = tabela_cargas.reset_index()

plt.scatter(tabela_cargas_chart['Fator 1'], tabela_cargas_chart['Fator 2'], s=30)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.05, point['y'], point['val'])

label_point(x = tabela_cargas_chart['Fator 1'],
            y = tabela_cargas_chart['Fator 2'],
            val = tabela_cargas_chart['index'],
            ax = plt.gca()) 

plt.axhline(y=0, color='black', ls='--')
plt.axvline(x=0, color='black', ls='--')
plt.ylim([-1.5,1.5])
plt.xlim([-1.5,1.5])
plt.title(f"{tabela_eigen.shape[0]} componentes principais que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=14)
plt.xlabel(f"PC 1: {round(tabela_eigen.iloc[0]['Variância']*100,2)}% de variância explicada", fontsize=14)
plt.ylabel(f"PC 2: {round(tabela_eigen.iloc[1]['Variância']*100,2)}% de variância explicada", fontsize=14)
plt.show()
   
plt.savefig('Factorial_Loadings_Plot.png')

# Gráfico da variância acumulada dos componentes principais
plt.figure(figsize=(12,8))

plt.title(f"{tabela_eigen.shape[0]} componentes principais que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=14)
ax = sns.barplot(x=tabela_eigen.index, y=tabela_eigen['Variância'], data=tabela_eigen, color='green')

ax.bar_label(ax.containers[0])
plt.xlabel("Componentes principais", fontsize=14)
plt.ylabel("Porcentagem de variância explicada (%)", fontsize=14)
plt.show()


# Calculando um Ranking, com base em : Predict Fator_x * Var. Fator_x
df['Ranking'] = 0

for index, item in enumerate(list(tabela_eigen.index)):
    variancia = tabela_eigen.loc[item]['Variância']
    
    df['Ranking'] = df['Ranking'] + df[tabela_eigen.index[index]]*variancia

    

