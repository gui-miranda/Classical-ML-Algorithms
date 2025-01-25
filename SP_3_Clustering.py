# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np


from sklearn.cluster import AgglomerativeClustering , KMeans

import scipy.stats as stats
import scipy.cluster.hierarchy as sch


""" 1. Ingestão dos Dados """
database = pd.read_csv("transformed_data.csv")  # dataframe com as principais features
df = database.drop("Unnamed: 0",axis=1)

columns = df.columns

""" 2. Algoritmos Não Supervisionados """

""" 2.1 Clustering """

# Normalizando as variaveis métricas através do Z-Score (Médio 0 e Desvio P = 1)

df_features = df[columns[1:-2]].sample(frac=1)

for col in df_features.columns:
    df_features.loc[:,col] = stats.zscore(df_features[col])


""" Encadeamento Hierarquico (utilizando a distância Euclidiana) : """ 

# A - Nearest Neighbor  - Dendograma
# plt.figure(figsize=(16,8))
# dendrogram = sch.dendrogram(sch.linkage(df_features, method = 'single', metric = 'euclidean')) #labels = list(df_features.index))
# plt.title('Dendrograma', fontsize=16)
# plt.xlabel('Observacoes', fontsize=16)
# plt.ylabel('Distância Euclidiana', fontsize=16)
# plt.axhline(y = 4.5, color = 'red', linestyle = '--')
# plt.show()



# Utilizando o Input do Dendograma com uma Proposta de 4 Clusters
cluster_sing = AgglomerativeClustering(n_clusters = 4, metric = 'euclidean', linkage = 'single')
indica_cluster_sing = cluster_sing.fit_predict(df_features)
df_features['cluster_single'] = indica_cluster_sing


cluster_comp = AgglomerativeClustering(n_clusters = 4, metric = 'euclidean', linkage = 'complete')
indica_cluster_comp = cluster_comp.fit_predict(df_features)
df_features['cluster_complete'] = indica_cluster_comp

cluster_avg = AgglomerativeClustering(n_clusters = 4, metric = 'euclidean', linkage = 'average')
indica_cluster_avg = cluster_avg.fit_predict(df_features)
df_features['cluster_average'] = indica_cluster_avg

gp_1 = df_features.groupby('cluster_single')['entropy_atomic_mass'].count()
gp_2 = df_features.groupby('cluster_complete')['entropy_atomic_mass'].count()
gp_3 = df_features.groupby('cluster_average')['entropy_atomic_mass'].count()

print(gp_1,gp_2,gp_3)


# Através da Análise dos Agrupamentos Hierarquicos, o método complete/average parecem ser os mais 
# adequados, oque aponta para uma alta homogeniedade das observações


""" Encadeamento Não Hierarquico :  K-Means """ 
# Iniciando com a estimativa inicial de 4 Clusters
df_features = df.set_index('material')[columns[1:-2]] 

for col in df_features.columns:
    df_features.loc[:,col] = stats.zscore(df_features[col])
    
kmeans = KMeans(n_clusters = 4, init = 'random').fit(df_features)
kmeans_clusters = kmeans.labels_


# Identificando as Centroides 
centroides = pd.DataFrame(kmeans.cluster_centers_)
centroides.columns = df_features.columns
centroides.index.name = 'cluster'
centroides

temp = df_features
temp['cluster_kmeans'] = kmeans_clusters
gp_4 = temp.groupby('cluster_kmeans')['entropy_atomic_mass'].count()
print(gp_4)


# Método Elbow para identificação do nº de clusters

## Elaborado com base na "inércia": distância de cada obervação para o centróide de seu cluster
## Quanto mais próximos entre si e do centróide, menor a inércia

inercias = []
K = range(1,50)#,df.shape[0])
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_features)
    inercias.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, inercias, 'bx-')
plt.axhline(y = 20, color = 'red', linestyle = '--')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Inércias', fontsize=16)
plt.title('Método do Elbow', fontsize=16)
plt.show()

# Adotando 5 Clusters como ponto Ótimo ! 
kmeans = KMeans(n_clusters = 5, init = 'random').fit(df_features)
kmeans_clusters = kmeans.labels_

# Construindo uma Análise de Variancia de um Fator ( TESTE F ) ! 
def teste_f_kmeans(kmeans, dataframe):
    
    variaveis = dataframe.columns

    centroides = pd.DataFrame(kmeans.cluster_centers_)
    centroides.columns = dataframe.columns
    centroides

    df = dataframe[variaveis]

    unique, counts = np.unique(kmeans.labels_, return_counts=True)

    dic = dict(zip(unique, counts))

    qnt_clusters = kmeans.n_clusters

    observacoes = len(kmeans.labels_)

    df['cluster'] = kmeans.labels_

    output = []

    for variavel in variaveis:

        dic_var={'variavel':variavel}

        # variabilidade entre os grupos

        variabilidade_entre_grupos = np.sum([dic[index]*np.square(observacao - df[variavel].mean()) for index, observacao in enumerate(centroides[variavel])])/(qnt_clusters - 1)

        dic_var['variabilidade_entre_grupos'] = variabilidade_entre_grupos

        variabilidade_dentro_dos_grupos = 0

        for grupo in unique:

            grupo = df[df.cluster == grupo]

            variabilidade_dentro_dos_grupos += np.sum([np.square(observacao - grupo[variavel].mean()) for observacao in grupo[variavel]])/(observacoes - qnt_clusters)

        dic_var['variabilidade_dentro_dos_grupos'] = variabilidade_dentro_dos_grupos

        dic_var['F'] =  dic_var['variabilidade_entre_grupos']/dic_var['variabilidade_dentro_dos_grupos']
        
        dic_var['sig F'] =  1 - stats.f.cdf(dic_var['F'], qnt_clusters - 1, observacoes - qnt_clusters)

        output.append(dic_var)

    df = pd.DataFrame(output)

    return df

output = teste_f_kmeans(kmeans,df_features)
kmeans_clusters = kmeans.labels_

# Conclusão : Aparentemente, todas as variaveis selecionadas mostraram sig Estatistica
# na construção de pelo menos 1 cluster.

# Analisando os Clusters Construidos e a Variável Alvo do Projeto "Temperatura Crítica"
temp_1 = df_features
temp_1['cluster_kmeans'] = kmeans_clusters
temp_1 = temp_1[['cluster_kmeans']]

temp_2 = df[['number_of_elements','critical_temp','material']].set_index("material")

df_merge = temp_1.join(temp_2, on='material')

# Realizando o Scatter Plot
plt.figure(figsize=(10,5))
sns.scatterplot(x='number_of_elements', y='critical_temp', data=df_merge, hue='cluster_kmeans', palette="viridis")
#sns.scatterplot(x='cluster_kmeans', y='critical_temp', data=df_merge)

plt.legend()
plt.show()