# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" 1. Ingestão dos Dados """
df_data   = pd.read_csv("train.csv") # dataframe com as princiapis features
df_labels = pd.read_csv("unique_m.csv")['material'] # dataframe indexando os materiais e numero de atomos de cada elemento

df = pd.concat([df_data,df_labels],axis=1).dropna(axis=1) # Removendo linhas nulas (1 ocorrencia)

print(df.dtypes)
