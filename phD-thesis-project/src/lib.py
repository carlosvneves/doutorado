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
    
     
    #  Unsample dos dados de Trim para Mensal
    upsampled = data_sa.resample('M')
    interpolated = upsampled.interpolate(method='linear')
    interpolated.tail(24)
    
    data = interpolated
    
    print('##'*25)
    print('# Matriz de Correlação #')        
    print(data.corr())
    print('##'*25)
    
    return data

 #%% Realiza previsão fora da amostra 
    def forecast(self, data, n_inputs, n_predictions, model):
            
    # =============================================================================
    #     model = load_model('model-4-n_endog,36-n_quarters,36-n_train_steps,5-n_features-300-n_nodes,100-n_epochs,32-n_batch.h5')
    #     print(model.summary())
    #     df = load_data()    
    #     df = df[['Inv', 'Agr', 'Ind', 'Inf', 'Com']] 
    # =============================================================================
        
        n_exog = data.shape[1]-1
        n_features = data.shape[1]
        ##############################################
        # Make forecasts
        #n_ahead = 12
        #n_before =24
         
    # =============================================================================
    #     inv = pd.Series(np.full(n_before,df.iloc[-n_before:]['Inv']))
    # # =============================================================================
    #     agro = pd.Series(np.full(n_before,np.mean(df.iloc[-n_before:]['Agr'])))
    #     ind = pd.Series(np.full(n_before,np.mean(df.iloc[-n_before:]['Ind'])))
    #     inf = pd.Series(np.full(n_before,np.mean(df.iloc[-n_before:]['Inf'])))
    #     com = pd.Series(np.full(n_before,np.mean(df.iloc[-n_before:]['Com'])))
    # #     
    # # =============================================================================
    # # =============================================================================
    # #     agro = pd.Series(np.zeros(n_before))
    # #     ind = pd.Series(np.zeros(n_before))
    # #     inf = pd.Series(np.zeros(n_before))
    # #     com = pd.Series(np.zeros(n_before))
    # # =============================================================================
    #     inv = pd.Series(np.zeros(n_before))
    #     
    #     
    #      
    #     
    #     df_forecast= pd.concat([inv,agro, ind, inf, com], axis=1)
    #     #dates_forecast = pd.date_range(start='2020-01-01', periods=n_before, freq='M')
    #     dates_forecast = pd.date_range(start=df.index[-n_before], periods=n_before, freq='M')
    #     df_forecast.index = pd.DatetimeIndex(dates_forecast)
    #     df_forecast.columns = df.columns
    #     
    #     strip = len(df) - n_before
    #     # ,df.iloc[-n_ahead:],df_forecast
    #     df_forecast = pd.concat((df.iloc[:strip],df_forecast),axis=0)
    # =============================================================================
            
        values = data.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        
        #n_quarters = 36
        
        n_features = data.shape[1]
        
        reframed = self.series_to_supervised(scaled, n_inputs, 1)
        
        values = reframed.values
    
        n_obs = n_inputs * n_features
        
        test_X = values[:, :n_obs]
        
        test_X = test_X.reshape((test_X.shape[0], n_inputs, n_features))
        
        # previsão por meio de batches
        pred_list = []
        y_unscaled = []
        n_steps=n_predictions
        
        for i in range(n_steps):
            batch = test_X[i].reshape((1, n_inputs, n_features))
            pred = model.predict(batch, verbose=1)
            batch = batch.reshape((1, n_inputs*n_features))
            inv_y = concatenate((np.array(pred), batch[:, -n_exog:]), axis=1)
            y_unscaled.append(np.array(inv_y))
            y = scaler.inverse_transform(inv_y)
            pred_list.append(y)
            batch = batch.reshape((1, n_inputs, n_features))
            batch = np.append(batch[:,1:,:], np.array(inv_y))
                
    
        yhat = np.array(pred_list)[:,:,0]
        
        add_dates = [data.index[-1] + pd.DateOffset(months=x) for x in range(0,n_steps+1) ]
        future_dates = pd.DataFrame(index=add_dates[1:],columns=data.columns)
        
        df_predict = pd.DataFrame(yhat,
                              index=future_dates[-n_steps:].index, columns=['Prediction'])
        
        df_proj = pd.concat([data,df_predict], axis=1)
       
        
        return df_proj, pred_list

