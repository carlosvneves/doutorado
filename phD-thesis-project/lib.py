#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:37:23 2020

@author: Carlos Eduardo Veras Neves
"""

#%% Importa Bibliotecas

# bibliotecas para as redes neurais
import tensorflow as tf
tf.keras.backend.clear_session()
from keras.layers.merge import concatenate
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard

# bibliotecas matemáticas
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler


# bilbiotecas de utilidades do sistema
import sys
import os
import pickle
from datetime import datetime
import time
from IPython import get_ipython

# As novas versões do Pandas e Matplotlib trazem diversas mensagens de aviso ao desenvolvedor. Vamos desativar isso.
# bibliotecas para visualização dos dados
import warnings
import matplotlib.cbook
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


# formatação dos gráficos
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.color'] = 'k'
from matplotlib.pylab import rcParams 
rcParams['figure.figsize'] = 25,15
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


import keras.backend as K


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = .98, allow_growth=True)
   
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)   

# Load the TensorBoard notebook extension.
get_ipython().run_line_magic('reload_ext', 'tensorboard')

OUTPUT_FLD = os.path.join('..','results')

def makedirs(fld):
	if not os.path.exists(fld):
		os.makedirs(fld)

#%% Carrega os dados
def load_data():
    
    # Carrega os dados
    github_repo = 'https://raw.githubusercontent.com/carlosvneves/doutorado/master/'
    desembolsos = pd.read_csv(github_repo + 'desembolsos.csv')
    pib = pd.read_csv(github_repo +'pib.csv')
    fbcf = pd.read_csv(github_repo +'fbcf.csv') 
    
    fbcf.index = pd.to_datetime(fbcf['date'])
    fbcf.drop(['date'],inplace=True, axis = 1)
    
       
    pib.index = pd.to_datetime(pib['date'])
    pib.drop(['date'],inplace=True, axis = 1)

    
    desembolsos.index = pd.to_datetime(desembolsos['date'])
    desembolsos.drop(['date'], inplace=True, axis = 1)
       
    data = desembolsos.groupby(pd.PeriodIndex(desembolsos.index, freq='Q')).mean()
    data = data.loc['1996Q1':]
    data.index = data.index.to_timestamp(freq='Q')
    
    for col in data.columns:
        data[col] = data[col]/pib['pib'].values * 100
       
       
    # Corte da série de acordo com a análise de tendência
    start = '2002Q1'
    
    data['Investimento'] = fbcf['fbcf'].values/pib['pib'].values *100
    data = data.loc[start:]
    
    print(data.describe())


    # Visualiza os dados originais
    data[['Investimento']].plot(figsize=(12,10));
    data[['Agropecuária','Indústria','Infraestrutura','Comércio e serviços', 'Total']].plot(figsize=(12,10));
    
    
    # Ajuste Sazonal (X-13 ARIMA) das séries para otimizar a modelagem
    X13_PATH = 'x13/'
    
    data_sa = pd.DataFrame(data)
    data_sa.rename(columns=lambda x: x[0:3], inplace=True)
    
    for col in data_sa:
        sa = sm.tsa.x13_arima_analysis(data_sa[col],x12path=X13_PATH)
        data_sa[col] = sa.seasadj.values
    
    data_sa.tail()
    
    
    # Visualiza dados com ajuste sazonal 
    
    data_sa[['Agr','Ind','Inf','Com','Tot']].plot(figsize=(12,8));
    data_sa['Inv'].plot(figsize=(12,8));
    
    
    # Prepara dados para modelo autorregressivo
        
    #data_lag1 = data_sa.shift(1).fillna(0)
    #data_lag2 = data_sa.shift(2).fillna(0)
    #data_lag3 = data_sa.shift(3).fillna(0)
    
       
    #df = pd.concat([data_sa,data_lag1['Inv'],data_lag2['Inv'],data_lag3['Inv']], axis=1, sort=False)
    #df.columns = ['Agr','Ind','Inf','Com','Tot','Inv','Inv-1','Inv-2','Inv-3']
    
       
    #  Unsample dos dados de Trim para Mensal
    upsampled = data_sa.resample('M')
    interpolated = upsampled.interpolate(method='spline', order=2)
    interpolated = upsampled.interpolate(method='linear')
    interpolated.tail(24)
    
    data = interpolated
    
    print('##'*25)
    print('# Matriz de Correlação #')        
    print(data.corr())
    print('##'*25)
    
    return data

#%% Configuração do modelo
# Lista de hiperparâmetros que serão testados
def config_model(n_exog = [4], n_steps = [8], n_train_steps = [24], n_features = [5],
                 n_nodes = [150,300],n_epochs = [100],n_batch = [128] ):
# forma de inserção manual dos dados: [[5],[8],[24],[5],[150,300],[50,100],[72]]
    
    configs = list()
    for i in n_exog:
        for j in n_steps:
            for k in n_train_steps:
                for l in n_features:
                    for m in n_nodes:
                        for n in n_epochs:
                          for o in n_batch:
                            cfg = [i,j,k, l, m, n, o]
                            configs.append(cfg)
                              
    print('\nTotal de Combinações de Hiperparâmetros: %d' % len(configs))
    return configs

