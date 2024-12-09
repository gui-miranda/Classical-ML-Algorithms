# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from tableone import TableOne
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pingouin as pg

""" 1. Ingestão dos Dados """
df_data = pd.read_csv("train.csv")  # dataframe com as principais features
df_labels = pd.read_csv("unique_m.csv")['material']  # dataframe indexando os materiais e número de átomos de cada elemento

df_merge = pd.concat([df_data, df_labels], axis=1).dropna(axis=1)  # Removendo linhas nulas (1 ocorrência)

# Corrigindo alguns tipos de dados de colunas
df_merge = df_merge.astype({col: 'float' for col in df_merge.select_dtypes('int64').columns})

# Organizando as colunas de interesse em um dicionário
columns_dict = {
    col.removeprefix('mean_'): (
                                col,
                                f"wtd_{col}", 
                                f"gmean_{col.removeprefix('mean_')}",
                                f"wtd_gmean_{col.removeprefix('mean_')}",
                                f"entropy_{col.removeprefix('mean_')}",
                                f"wtd_entropy_{col.removeprefix('mean_')}",
                                f"wtd_std_entropy_{col.removeprefix('mean_')}",
                                f"wtd_std_gmean_{col.removeprefix('mean_')}",
                                f"wtd_std_mean_{col.removeprefix('mean_')}"
                                ) 
    for col in df_merge.columns if col.startswith('mean_')
}

df = df_merge

""" ------------------- """



""" 2. Análise Exploratória"""
""" 2.1 Histogram Plots """
plt.figure(figsize=(20,15))
plt.subplot(3,3,1)
plt.title("CriticalTemperature")
plt.xlabel(" ")
sns.histplot(df['critical_temp'],label='critical_temp', kde=True)

for k,(title,cols) in enumerate(columns_dict.items(), start=2):
    plt.subplot(3,3,k)    
    plt.title(f'{title}')
    plt.xlabel(" ")
    
    sns.set_style("ticks")
    sns.histplot(df[cols[0]],label='mean', kde=True)
    sns.histplot(df[cols[1]],label='wtd_mean', kde=True)
    sns.histplot(df[cols[2]],label='gmean', kde=True)
    sns.histplot(df[cols[3]],label='wtd_gmean', kde=True)
    plt.legend()
plt.savefig('Histplot_Means_Gmeans.png')


plt.figure(figsize=(20,15))
plt.subplot(3,3,1)
plt.title("NumberOfElements")
plt.xlabel(" ")
sns.histplot(df['number_of_elements'],label='number_of_elements', kde=True)
for k,(title,cols) in enumerate(columns_dict.items(), start=2):
    plt.subplot(3,3,k)    
    plt.title(f'{title}')
    plt.xlabel(" ")
    
    sns.set_style("ticks")
    sns.histplot(df[cols[4]],label='entropy', kde=True)
    sns.histplot(df[cols[5]],label='wtd_entropy', kde=True)
 
    plt.legend()
plt.savefig('Histplot_Entropy_Measures.png')

""" 2.2 Correlation Matrix """
corr = df.drop("material", axis=1).corr(method='pearson')

f = plt.figure(figsize=(20, 15))
cax = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
f.colorbar(cax)

plt.title("Correlation Matrix")
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)

plt.savefig('Correlation_Matrix.png')

temp = corr.reset_index()[['index','critical_temp']]
temp = temp[ ( (temp['critical_temp'] >= 0.5) & (temp['critical_temp'] < 1) ) | (temp['critical_temp'] <= -0.5) ]

features_cols = temp['index'].values.tolist()


""" 2.2 Outliers Analysis """

df_quartiles = pd.DataFrame()
q1,q2,q3 = [],[],[]
for col in features_cols :
    quartiles = df[col].quantile([0.25,0.5,0.75]).tolist()
    q1.append(quartiles[0])
    q2.append(quartiles[1]) 
    q3.append(quartiles[2])

df_quartiles['kpi'] = features_cols
df_quartiles['q1'] = q1
df_quartiles['q2'] = q2
df_quartiles['q3'] = q3
df_quartiles['interquartile_distance'] = df_quartiles['q3']-df_quartiles['q1']
df_quartiles['trashold_top']    = df_quartiles['q3'] + (1.5*df_quartiles['interquartile_distance'] )
df_quartiles['trashold_bottom'] = df_quartiles['q1'] - (1.5*df_quartiles['interquartile_distance'] )

temp = df[features_cols]
qtd_outliers = []
for col in features_cols :
    top    = df_quartiles[df_quartiles['kpi']==col]['trashold_top'].values[0]
    bottom = df_quartiles[df_quartiles['kpi']==col]['trashold_bottom'].values[0]
    
    outliers = temp.query(f"({col} >= {top}) or (({col} <= {bottom}))").shape[0]
    qtd_outliers.append(outliers)
    
df_quartiles['qtd_outliers'] = qtd_outliers  
df_quartiles['percent_outliers'] =round((df_quartiles['qtd_outliers']/df.shape[0])*100,2)  
df_quartiles = df_quartiles.sort_values(by = 'qtd_outliers',ascending = False).reset_index().drop('index',axis=1)


""" 3. Algoritmos Não Supervisionados """

""" 3.1 Redução de Dimensionalidade - PCA """

features_cols.extend(["critical_temp","material"])
df = df[features_cols]


# Testando a adequação e possibilidade de extração de componentes principais
# nesse conjunto de dados, através do Teste de Bartllet.
# O Teste de Bartllet avalia a hipótese de que as amostras do conjunto possuem variância iguais.
# H0 : Todas as Variâncias são iguais na amostra (Não há FATORES). 
# H1 : Pelo menos 2 das amostras possuem variância distinta (Há FATORES)
# Se p_value < 5%, REJEITA-SE H0

df_features = df[features_cols[0:-2]]
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

print(tabela_cargas)



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
     
# 


    

