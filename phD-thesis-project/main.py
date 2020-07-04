#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:16:29 2020

@author: Carlos Eduardo Veras Neves - PhD candidate
University of Brasilia - UnB
Department of Economics - Applied Economics

Thesis Supervisor: Geovana Lorena Bertussi, PhD.
Title:Professor of Department of Economics - UnB
    
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


#%% Modelo de Redes Neurais Recorrentes (RNN)

#%% Conversão da estrutura de séries para aprendizado supervisionado
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


#%%Mede performance do modelo
def performance(y_true, y_pred): 
        mse = mean_squared_error(y_true,y_pred)
        mape = mean_squared_error(y_true,y_pred)
        print('MSE das previsões é {}'.format(round(mse, 2))+
                      '\nRMSE das previsões é {}'.format(round(np.sqrt(mse), 2))+
                      '\nMAPE das previsões é {}'.format(round(mape, 2)))
        return mse


#%%Treino da RNA
def train_model(data, cfg):

  n_endog,n_steps, n_train_steps, n_features, n_nodes, n_epochs, n_batch = cfg

  values = data.values
  values = values.astype('float32')
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)
  # specify the number of lag quarters
  # frame as supervised learning
  reframed = series_to_supervised(scaled, n_steps, 1)

  # split into train and test sets
  values = reframed.values
  
  train = values[:n_train_steps, :]
  test = values[n_train_steps:, :]
  
  n_obs = n_steps * n_features
  #n_obs = n_steps * n_endog    

  # O MODELO É AUTORREGRESSIVO (ALTERAR?)  
  train_X, train_y = train[:, :n_obs], train[:, -n_features]
  test_X, test_y = test[:, :n_obs], test[:, -n_features]
  #train_X, train_y = train[:, :n_obs], train[:, -n_endog]
  #test_X, test_y = test[:, :n_obs], test[:, -n_endog]
  
  #print(train_X.shape[0])
  # reshape input to be 3D [samples, timesteps, features]
  train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
  test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
  #train_X = train_X.reshape((train_X.shape[0], n_steps, n_endog))
  #test_X = test_X.reshape((test_X.shape[0], n_steps, n_endog))  
  # Criação e treinamento do modelo LSTM Padrão
  #n_input = n_steps*n_features
  #n_input = n_steps*n_endog
  n_neurons = n_nodes
  batch_size=n_batch


  # reshape input to be 3D [samples, timesteps, features]
  #train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
  #test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
  # Create a MirroredStrategy.
  #strategy = tf.distribute.MirroredStrategy()
  #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  # Open a strategy scope.
  #with strategy.scope():

  #número de neurônios nas camadas ocultas
  hidden_nodes = int(n_nodes*2/3)
  dropout = 0.2
  # modelo de rede de acordo com a configuração
  model = keras.Sequential()

  # CUDNN LSTM implementation  
  model.add(keras.layers.LSTM(units = n_neurons, activation = 'tanh',recurrent_activation = 'sigmoid',
                    recurrent_dropout = 0,unroll = False, use_bias = True,
                    input_shape=(train_X.shape[1], train_X.shape[2])))
  model.add(keras.layers.Dropout(dropout))
  model.add(keras.layers.Dense(hidden_nodes, activation = 'relu'))
  model.add(keras.layers.Dropout(dropout))
  model.add(keras.layers.Dense(1))
  
  # Stacked LSTM model
# =============================================================================
#       model.add(keras.layers.LSTM(units = n_neurons, activation = 'relu',recurrent_activation = 'sigmoid',
#                         recurrent_dropout = 0,unroll = False, use_bias = True, return_sequences = True,
#                         input_shape=(train_X.shape[1], train_X.shape[2])))
#       # adicionar camada LSTM para usar o disposito de recorrência
#       model.add(keras.layers.LSTM(units = n_neurons, activation = 'relu'))
#       model.add(keras.layers.Dropout(dropout))
#       model.add(keras.layers.Dense(hidden_nodes, activation = 'tanh'))
#       model.add(keras.layers.Dropout(dropout))
#       model.add(keras.layers.Dense(1))
#       
# =============================================================================
  
  
  
  
  learning_rate=1.0e-3
  
  #session = K.get_session()
  #weights_initializer = tf.compat.v1.variables_initializer(layer.weights)
  #session.run(weights_initializer)  
  
  opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

  model.compile(loss="mse", optimizer=opt, metrics=["mae"])

  tensorboard = TensorBoard(log_dir="logs/{}-{}-{}-{}-".format(n_features, 
                                                               n_nodes, n_epochs, n_batch) + datetime.now().strftime("%Y%m%d-%H%M%S"))  
  
  patience_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

  # fit network
  model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, 
            validation_data=(test_X, test_y), verbose=0, shuffle=False, 
            callbacks=[patience_callback, tensorboard])
  
# =============================================================================
#       model.save("model{}-{}-{}-{}-{}.h5".format(n_features, n_nodes, n_epochs, 
#                                                    n_batch,datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))  
# =============================================================================
  
  
  
  return model, test_X, test_y,scaler


#%% Avaliação do Modelo
def eval_model(data, cfg, n_rep=10):
  
  n_endog,n_steps, n_train_steps, n_features, n_nodes, n_epochs, n_batch = cfg

  resultado = []
  perf = np.zeros((n_rep))

  # Loop (TODO)
  # Vamos repetir o processo de treinamento por 20 vezes e armazenar todos os resultados, pois assim usaremos
  # diferentes amostras. Ao final, tiramos a média para encontrar as previsões. 
  # make a prediction
  print('\n')
  print('##'*35)
  series_par = "{}-n_endog,{}-n_steps,{}-n_train_steps,{}-n_features".format(n_endog,
                                                                                   n_steps, 
                                                                                   n_train_steps, 
                                                                                   n_features)
  
  model_par =  "{}-n_nodes,{}-n_epochs,{}-n_batch".format(n_nodes, n_epochs, n_batch)                                                                                                          
                                                                                                              
  
  print(f'## Avaliação do Modelo: \n{series_par}\n{model_par}\n ## ')
  print(datetime.now().strftime("%Y/%m/%d-%H:%M:%S\n"))
  
  for i in range(n_rep):
    
    model, testx, testy, scaler = train_model(data, cfg)
    #modelo.model.model
    test_x = testx
    test_y = testy

    yhat = model.predict(test_x)
    #print(yhat.shape)
    test_x = test_x.reshape((test_x.shape[0], n_steps*n_features))
    
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_x[:, -n_endog:]), axis=1)##
    yhat = scaler.inverse_transform(inv_yhat)
    yhat = yhat[:,0]

    
    resultado.append(yhat)

    print(f'\nRepetição:{i+1}')
    print(f'# épocas:({n_epochs}) # neurônios:({n_nodes}) # batch:({n_batch})')
    #print(f'loss:{round(modelo.history.history["loss"][-1],4)} - end val_loss: {round(modelo.history.history["val_loss"][-1],4)}\n')


    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_x[:, -n_endog:]), axis=1)
    y = scaler.inverse_transform(inv_y)
    y = y[:,0]
    
    perf[i] = performance(y,yhat)

  perf_mean = np.mean(perf)
  resultado = np.array(resultado) 
  
  # Loop para gerar as previsões finais
  result_mean = np.zeros((resultado.shape[1],1))
  for i in range(resultado.shape[1]):
    result_mean[i] = np.mean(resultado[:,i])
  
  model.save("models/model-pct-{}-{}.h5".format(series_par,model_par))  
  #,datetime.now().strftime("%Y%m%d-%H%M%S")
  #model.summary()
    
  return result_mean, perf_mean, cfg
  


#%%Grid Search
# Função para o Grid Search
def grid_search(data, cfg_list):
    errors = []
    results = []
    hyperparams = []
    #resultado = avalia_modelo(modelo, test_X, config[0])
    # Gera os scores
    for cfg in cfg_list:
      result,error,hyper = eval_model(data, cfg) 
      results.append(result)
      errors.append(error)
      hyperparams.append(hyper)

    # Ordena os hiperparâmetros pelo erro
    #errors.sort(key = lambda tup: tup[1])
    return results,errors, hyperparams


#%%Seleciona o melhor resultado
def best_result(error, result, hyperparms):
  index = error.index(min(error))
  best_fit = result[index]
  best_hyper = hyperparms[index]
  print(f"Parâmetros do melhor modelo:{best_hyper}")
       
  print(f"Menor MSE: {round(min(error),3)}")
  return best_fit, best_hyper


#%% Execução do Modelo
def run_model(data,cfg):
  start = time.time()


  results, error, hyperparams = grid_search(data, cfg)

  print(f"MSE: ({min(error)})\n")


  end = time.time()
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
  
  return results, error, hyperparams


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
    
    print('##*25')
    print('# Matriz de Correlação #')        
    print(data.corr())
    print('##*25')
    
    return data

#%% Realiza previsão fora da amostra 
def forecast(data, n_inputs, n_predictions, model):
        
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
    
    reframed = series_to_supervised(scaled, n_inputs, 1)
    
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

#%% Carrega dados do melhor modelo
def load_best_model():
    
    with open ('best_lstm_par.pkl', 'rb') as fp:
        best_par = pickle.load(fp)
   
    best_file = 'models/model-pct-{}-n_endog,{}-n_steps,{}-n_train_steps,{}-n_features-{}-n_nodes,{}-n_epochs,{}-n_batch.h5'.format(best_par[0],
                                                                                                                                    best_par[1],
                                                                                                                                    best_par[2],
                                                                                                                                    best_par[3],
                                                                                                                                    best_par[4],
                                                                                                                                    best_par[5],
                                                                                                                                    best_par[6])
    
    #=========================================================================
    best_model = keras.models.load_model(best_file)    
    
    with open ('best_lstm_res.pkl', 'rb') as fp:
        best_res = pickle.load(fp)
    
         
    return best_model, best_res, best_par[1]

#%% Função que simula modelo LSTM
def LSTM(data):
    from numpy.random import seed
    seed(1)
       
    # Load the TensorBoard notebook extension.
    get_ipython().run_line_magic('reload_ext', 'tensorboard')
    
    df = data
        
    var = ['Inv', 'Agr', 'Ind', 'Inf', 'Com']
    var_ar = ['Inv', 'Agr', 'Ind', 'Inf', 'Com','Inv-1']


    #%% Modelo LSTM - sem termos Autorregressivos %
    config = config_model(n_steps = [24,36],n_train_steps = [24,36],n_batch = [16,32])
    results_lstm, error_lstm, hyperparams_lstm = run_model(df[var], config)
    # armazena o melhor resultado   
    best_lstm_res, best_lstm_par = best_result(error_lstm, results_lstm, hyperparams_lstm)
    
    # Escreve os resultados em arquivo
      
    with open('best_lstm_res.pkl', 'wb') as fp:
        pickle.dump(best_lstm_res, fp) 
    
    
    with open('best_lstm_par.pkl', 'wb') as fp:
        pickle.dump(best_lstm_par, fp)  
    
    
    #%% # Visualiza a previsão do modelo - dados de teste com menor MSE
    # Plot
    plt.figure()
    
    # Série original
    plt.plot(df.index, 
              df['Inv'].values,
              label = 'Valores Observados',
              color = 'Red')
    plt.plot(df.index[-len(best_lstm_res):], 
             best_lstm_res,
             label = 'Previsões com Modelo de Redes Neurais LSTM', 
             color = 'Black')
    plt.title('Previsões com Modelo de Redes Neurais Recorrentes')
    plt.xlabel('Ano')
    plt.ylabel('Investimento (%PIB)')
    plt.legend()
    plt.show()	


#%% Função principal
def main():
    # Versões dos pacotes usados neste jupyter notebook
    get_ipython().run_line_magic('reload_ext', 'watermark')
    get_ipython().run_line_magic('watermark', '-a "Carlos Eduardo Veras Neves" --iversions')
        
    # Carrega os dados para simulação
    df = load_data()
    
    # Simula o modelo LSTM
    LSTM(df)
    
    # Carrega o modelo que apresentou o melhor resultado na simulação
    best_lstm_model, best_lstm_res, n_inputs = load_best_model()
    
    df = df[['Inv', 'Agr', 'Ind', 'Inf', 'Com']]
    
    # Cria Data Frame a partir da melhor previsão dentro da amostra
    df_lstm = pd.DataFrame(best_lstm_res, columns=['Inv_lstm'])
    df_lstm.index = df.index[-len(df_lstm):]  
    
    # Realiza previsão fora da amostra
    df_proj, pred_list = forecast(df, n_inputs, 24, best_lstm_model)        
    
    # Cria Data Frame com todos os resultados    
    df_proj = pd.concat([df_proj,df_lstm], axis=1)

    
    # Gráfico com a série original, previsão do modelo dentro da amostra
    # e previsão do modelo fora da amostra
    plt.figure()
    plt.plot(df_proj.index, df_proj['Inv'],
             label = 'Valores Observados')
    plt.plot(df_proj.index, df_proj['Inv_lstm'], 
             label = 'Rede Neural - dentro da amostra',color='black')
    plt.plot(df_proj.index, df_proj['Prediction'], 
             label = 'Rede Neural - fora da amostra',color='blue')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.title('Previsões com Modelo de Redes Neurais Recorrentes')
    plt.xlabel('Ano')
    plt.ylabel('Investimento (%PIB)')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
	main()
    # TODO: inserir no nome do modelo n_time_steps;n_train_steps (mudar o nome das variáveis n_quarters, n_train_steps)


    
     

    