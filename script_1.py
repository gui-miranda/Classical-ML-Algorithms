# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" 1. Ingestão dos Dados """
df_data = pd.read_csv("train.csv")  # dataframe com as principais features
df_labels = pd.read_csv("unique_m.csv")['material']  # dataframe indexando os materiais e número de átomos de cada elemento

df_merge = pd.concat([df_data, df_labels], axis=1).dropna(axis=1)  # Removendo linhas nulas (1 ocorrência)

# Corrigindo alguns tipos de dados de colunas
df_merge = df_merge.astype({col: 'float' for col in df_merge.select_dtypes('int64').columns})

# Organizando as colunas de interesse em um dicionário
columns_dict = {
    col.removeprefix('mean_'): (col,
                                f"wtd_{col}", 
                                f"gmean_{col.removeprefix('mean_')}",
                                f"wtd_gmean_{col.removeprefix('mean_')}",
                                f"entropy_{col.removeprefix('mean_')}",
                                f"wtd_entropy_{col.removeprefix('mean_')}"
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

