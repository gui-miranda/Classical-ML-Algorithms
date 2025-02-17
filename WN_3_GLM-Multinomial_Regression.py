import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from scipy import stats # estatística chi2
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from ucimlrepo import fetch_ucirepo 
import warnings

""" 1. Data Aquisition """
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 


df = wine_quality.data.original
df['quality'] = df['quality'] - np.min(df['quality'])

df.loc[(df['quality'] == 0) | (df['quality'] == 1),'quality'] = 1
df.loc[(df['quality'] == 5) | (df['quality'] == 6),'quality'] = 5
df['quality'] = df['quality']-1


""" 2.1 Multinomial Regression - volatile_acidity """
from statsmodels.discrete.discrete_model import MNLogit

x = df[df.columns[0:-2]]
y = df['quality']

X = sm.add_constant(x)

modelo_atrasado = MNLogit(endog=y, exog=X).fit()

modelo_atrasado.summary()

def Qui2(modelo_multinomial):
    maximo = modelo_multinomial.llf
    minimo = modelo_multinomial.llnull
    qui2 = -2*(minimo - maximo)
    pvalue = stats.distributions.chi2.sf(qui2,1)
    df = pd.DataFrame({'Qui quadrado':[qui2],
                       'pvalue':[pvalue]})
    return df

Qui2(modelo_atrasado)

phats = pd.DataFrame(modelo_atrasado.predict())

df_final = pd.concat([df, phats], axis=1)

classes_predict = phats.idxmax(axis=1)
df_final['y_predict'] = classes_predict

#Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_final,
                       index=['y_predict'],
                       columns=['quality'],
                       aggfunc='size')

table = table.reindex(range(5), fill_value=0).fillna(0)

# Função para calcular as métricas
def calculate_metrics(confusion_matrix):
    # Acurácia
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix.values)
    
    # Precisão, Recall e F1-Score por classe
    precision = {}
    recall = {}
    f1_score = {}
    
    for cls in confusion_matrix.index:
        true_positives = confusion_matrix.loc[cls, cls]
        false_positives = np.sum(confusion_matrix[cls]) - true_positives
        false_negatives = np.sum(confusion_matrix.loc[cls]) - true_positives
        
        precision[cls] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall[cls] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1_score[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) != 0 else 0
    
    # Média das métricas
    avg_precision = np.mean(list(precision.values()))
    avg_recall = np.mean(list(recall.values()))
    avg_f1_score = np.mean(list(f1_score.values()))
    
    print(f"Acurácia: {accuracy:.2f}")
    print(f"Precisão Média: {avg_precision:.2f}")
    print(f"Recall Médio: {avg_recall:.2f}")
    print(f"F1-Score Médio: {avg_f1_score:.2f}")
    
    return accuracy, avg_precision, avg_recall, avg_f1_score

# Calcular as métricas
accuracy, avg_precision, avg_recall, avg_f1_score = calculate_metrics(table)

