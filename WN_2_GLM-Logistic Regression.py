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
df = df.drop("quality",axis=1)

df.loc[df['color'] == 'red', 'color'] = 0
df.loc[df['color'] == 'white', 'color'] = 1
df['color'] = df['color'].astype('int32')


""" 2. Regressão Linear - Simples - Usanod Varivel volatile_acidity"""

temp = df.copy()
modelo_1 = smf.glm(formula='color ~ volatile_acidity', data=temp,
                         family=sm.families.Binomial()).fit()

#Parâmetros do modelo
modelo_1.summary()

temp['phat'] = modelo_1.predict()

plt.figure(figsize=(15,8))
plt.title("Fited Logistic Regression")
plt.scatter(temp['volatile_acidity'],temp['phat'])
plt.xlabel('Volatity Acidity')
plt.ylabel('Predict Prob')
plt.show()