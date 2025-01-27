# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 

# variable information 
#print(wine_quality.variables) 

df = wine_quality.data.original

df.loc[df['color'] == 'red', 'color'] = 0
df.loc[df['color'] == 'white', 'color'] = 1
df['color'] = df['color'].astype('int32')

""" ------------------- """


""" 2. Análise Exploratória"""
""" 2.1 Histogram Plots """
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title("Quality Variable Distribution")
plt.xlabel(" ")
sns.histplot(df['quality'],label='quality', kde=True)

plt.subplot(1,2,2)
plt.title("Color Variable Distribution")
plt.xlabel(" ")
sns.histplot(df['color'],label='color')
plt.show()

plt.figure(figsize=(20,8))
i = 1
for col in df.columns[:-2]:
    plt.subplot(2,5,i)    
    plt.title(f'{col}')
    plt.xlabel(" ")
    
    sns.set_style("ticks")
    sns.histplot(df[col], kde=True)

    i += 1

""" 2.2 Correlation Matrix """
corr = df.corr(method='pearson')

f = plt.figure(figsize=(20, 15))
cax = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
f.colorbar(cax)

plt.title("Correlation Matrix")
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
plt.show()

print("Correlações Variavel 'Quality'")
print(corr['quality'])

print("")
print("Correlações Variavel 'Color'")
print(corr['color'])

# Criando Dummies para a Varivel Target : Quality
dummies = pd.get_dummies(df['quality']).rename(columns=lambda x: 'quality_' + str(x))
dummies_names = dummies.columns
temp = pd.concat([df, dummies], axis=1).drop('quality', axis=1)
corr = temp.corr(method='pearson')
corr_dumies = corr[dummies_names][0:11]


""" 2.3 Outliers Analysis """

df_quartiles = pd.DataFrame()
q1,q2,q3 = [],[],[]
for col in df.columns[:-2] :
    quartiles = df[col].quantile([0.25,0.5,0.75]).tolist()
    q1.append(quartiles[0])
    q2.append(quartiles[1]) 
    q3.append(quartiles[2])

df_quartiles['kpi'] = df.columns[:-2]
df_quartiles['q1'] = q1
df_quartiles['q2'] = q2
df_quartiles['q3'] = q3
df_quartiles['interquartile_distance'] = df_quartiles['q3']-df_quartiles['q1']
df_quartiles['trashold_top']    = df_quartiles['q3'] + (1.5*df_quartiles['interquartile_distance'] )
df_quartiles['trashold_bottom'] = df_quartiles['q1'] - (1.5*df_quartiles['interquartile_distance'] )

temp = df[df.columns[:-2]]
qtd_outliers = []
for col in df.columns[:-2] :
    top    = df_quartiles[df_quartiles['kpi']==col]['trashold_top'].values[0]
    bottom = df_quartiles[df_quartiles['kpi']==col]['trashold_bottom'].values[0]
    
    outliers = temp.query(f"({col} >= {top}) or (({col} <= {bottom}))").shape[0]
    qtd_outliers.append(outliers)
    
df_quartiles['qtd_outliers'] = qtd_outliers  
df_quartiles['percent_outliers'] =round((df_quartiles['qtd_outliers']/df.shape[0])*100,2)  
df_quartiles = df_quartiles.sort_values(by = 'qtd_outliers',ascending = False).reset_index().drop('index',axis=1)


# """ 2.2 Correlation Matrix """
# corr = df.drop("material", axis=1).corr(method='pearson')

# f = plt.figure(figsize=(20, 15))
# cax = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
# f.colorbar(cax)

# plt.title("Correlation Matrix")
# plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)

# plt.savefig('Correlation_Matrix.png')

# temp = corr.reset_index()[['index','critical_temp']]
# temp = temp[ ( (temp['critical_temp'] >= 0.5) & (temp['critical_temp'] < 1) ) | (temp['critical_temp'] <= -0.5) ]

# features_cols = temp['index'].values.tolist()
# print(features_cols)


# """ 2.2 Outliers Analysis """

# df_quartiles = pd.DataFrame()
# q1,q2,q3 = [],[],[]
# for col in features_cols :
#     quartiles = df[col].quantile([0.25,0.5,0.75]).tolist()
#     q1.append(quartiles[0])
#     q2.append(quartiles[1]) 
#     q3.append(quartiles[2])

# df_quartiles['kpi'] = features_cols
# df_quartiles['q1'] = q1
# df_quartiles['q2'] = q2
# df_quartiles['q3'] = q3
# df_quartiles['interquartile_distance'] = df_quartiles['q3']-df_quartiles['q1']
# df_quartiles['trashold_top']    = df_quartiles['q3'] + (1.5*df_quartiles['interquartile_distance'] )
# df_quartiles['trashold_bottom'] = df_quartiles['q1'] - (1.5*df_quartiles['interquartile_distance'] )

# temp = df[features_cols]
# qtd_outliers = []
# for col in features_cols :
#     top    = df_quartiles[df_quartiles['kpi']==col]['trashold_top'].values[0]
#     bottom = df_quartiles[df_quartiles['kpi']==col]['trashold_bottom'].values[0]
    
#     outliers = temp.query(f"({col} >= {top}) or (({col} <= {bottom}))").shape[0]
#     qtd_outliers.append(outliers)
    
# df_quartiles['qtd_outliers'] = qtd_outliers  
# df_quartiles['percent_outliers'] =round((df_quartiles['qtd_outliers']/df.shape[0])*100,2)  
# df_quartiles = df_quartiles.sort_values(by = 'qtd_outliers',ascending = False).reset_index().drop('index',axis=1)

# features_cols.extend(["critical_temp","material"])
# df[features_cols].to_csv("transformed_data.csv")
