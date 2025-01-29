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
warnings.filterwarnings('ignore')

""" 1. Data Aquisition """
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 


df = wine_quality.data.original

df.loc[df['color'] == 'red', 'color'] = 0
df.loc[df['color'] == 'white', 'color'] = 1
df['color'] = df['color'].astype('int32')


""" 2. Regressão Linear - Simples - Usanod Varivel volatile_acidity"""

temp = df.copy().drop("quality",axis=1)
modelo_1 = smf.glm(formula='color ~ volatile_acidity', data=temp,
                         family=sm.families.Binomial()).fit(disp=False)

#Parâmetros do modelo
modelo_1.summary()

#Plotando os resultados Obtidos
temp['phat'] = modelo_1.predict()

plt.figure(figsize=(8,6))
plt.title("Fited Logistic Regression")
plt.scatter(temp['volatile_acidity'],temp['phat'],color='black')
plt.scatter(temp['volatile_acidity'],temp['color'],color='orange',label='True Values')
plt.xlabel('Volatity Acidity')
plt.ylabel('Predict Prob')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)  # Grade para melhorar a visualização
plt.tight_layout()  # Ajusta o layout para evitar cortes
plt.show()

"""2.1 Definição do Cutoff"""
# Definando o Cuttoff com base na ROC-Curve e Indice de Youden
from sklearn import metrics

roc_curve = metrics.roc_curve(temp['color'],temp['phat'])
fpr, tpr, thresholds = metrics.roc_curve(temp['color'],  temp['phat'])

auc = metrics.roc_auc_score(temp['color'],  temp['phat'])

# Youden Index -> Max(J) = Max(True Positives - False Positives) - Calibra Sensibilidade e Especificidade
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_cutoff = thresholds[optimal_idx]
print(f"Cutoff ideal: {optimal_cutoff}")

optimal_fpr = fpr[optimal_idx]
optimal_tpr = tpr[optimal_idx]

# Configurações do plot
plt.figure(figsize=(8, 6))  # Tamanho adequado para publicação

plt.scatter(optimal_fpr, optimal_tpr, color='orange', marker='o', s=100, 
            label=f'Optimal Cutoff (Threshold = {optimal_cutoff:.2f})') # Adiciona o ponto de máximo Índice de Youden 

plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {auc:.2f})',color='black')  # Linha da curva ROC
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')  # Linha de referência

plt.xlabel('False Positive Rate (FPR)', fontsize=12)  # Rótulo do eixo X
plt.ylabel('True Positive Rate (TPR)', fontsize=12)  # Rótulo do eixo Y
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)  # Título do gráfico

plt.legend(loc='lower right', fontsize=12)  # Legenda
plt.grid(True, linestyle='--', alpha=0.6)  # Grade para melhorar a visualização
plt.tight_layout()  # Ajusta o layout para evitar cortes
plt.show()

"""2.1 Analisando a Matriz de Confusão"""
from sklearn import metrics 

temp['y_predict'] = temp['phat']
temp.loc[temp['phat'] < optimal_cutoff ,'y_predict'] = 0
temp.loc[temp['phat'] >= optimal_cutoff,'y_predict'] = 1

cm = metrics.confusion_matrix(temp['color'], temp['y_predict'])

plt.figure(figsize=(8, 6))  # Tamanho adequado para publicação
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])  # Rótulos das classes
disp.plot(cmap='Blues', values_format='d')  # Formato dos valores e mapa de cores

plt.title('Matriz de Confusão', fontsize=14, pad=20)  
plt.xlabel('Classe Predita', fontsize=12) 
plt.ylabel('Classe Verdadeira', fontsize=12)  
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)  
plt.gca().invert_xaxis()  
plt.gca().invert_yaxis()  
plt.grid(False)  
plt.tight_layout()  

accuracy  = metrics.accuracy_score(temp['color'],temp['y_predict'])
f1_score  = metrics.f1_score(temp['color'],temp['y_predict'])
recall    = metrics.recall_score(temp['color'],temp['y_predict'])
precision = metrics.precision_score(temp['color'],temp['y_predict'])

print(f"Accuracy : {accuracy:.2f} \nF1-Score : {f1_score:.2f} \nRecall(TP/(TP+FN)) : {recall:.2f}\nPrecision(TP/(TP+FP)) : {precision:.2f}")



""" 2.2 Regressão Logistica Simples - Cross Validation"""

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score,recall_score,f1_score,precision_score,confusion_matrix

temp = df.drop("quality",axis=1).copy()

#Seprando os conjuntos de treino e teste
X = temp[temp.columns[0:11]]
y = temp['color']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estimando o Modelo
model = LogisticRegression()

# Utilizando uma validação cruzada
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')

print("Média do AUC-ROC:", np.mean(cv_scores))

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva
y_pred = model.predict(X_test)  # Previsões das classes

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))  # Tamanho adequado para publicação
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])  # Rótulos das classes
disp.plot(cmap='Blues', values_format='d')  # Formato dos valores e mapa de cores

plt.title('Matriz de Confusão', fontsize=14, pad=20)  
plt.xlabel('Classe Predita', fontsize=12) 
plt.ylabel('Classe Verdadeira', fontsize=12)  
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)  
plt.gca().invert_xaxis()  
plt.gca().invert_yaxis()  
plt.grid(False)  
plt.tight_layout()  


auc     = roc_auc_score(y_test, y_pred_proba)
accuracy  = accuracy_score(y_test, y_pred)
f1_score  = f1_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(f"ROC-AUC : {auc:.2f} \nAccuracy : {accuracy:.2f} \nF1-Score : {f1_score:.2f} \nRecall(TP/(TP+FN)) : {recall:.2f}\nPrecision(TP/(TP+FP)) : {precision:.2f}")
