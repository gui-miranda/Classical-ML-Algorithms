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
print(features_cols)


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

features_cols.extend(["critical_temp","material"])
df[features_cols].to_csv("transformed_data.csv")
