#!/usr/bin/env python
# coding: utf-8

# # Importa Bibliotecas

# In[1]:

from IPython import get_ipython
# multivariate mlp example
import tensorflow as tf
tf.keras.backend.clear_session()


from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.layers import LSTM#,CuDNNLSTM
from keras.layers import Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import GRU
from keras.layers import InputLayer
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Reshape

#from keras.callbacks import EarlyStopping
#from keras.callbacks import TensorBoard
#from keras.callbacks import TensorBoard
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard


#import autokeras as ak
from sklearn.model_selection import train_test_split


from numpy import array
from numpy import hstack
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm
from matplotlib import pyplot


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


from pandas import DataFrame
from pandas import concat
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import MinMaxScaler

# As novas versões do Pandas e Matplotlib trazem diversas mensagens de aviso ao desenvolvedor. Vamos desativar isso.
import sys
import os
from datetime import datetime

import warnings
import matplotlib.cbook
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


# Imports para formatação dos gráficos
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.color'] = 'k'
from matplotlib.pylab import rcParams 
rcParams['figure.figsize'] = 15,7
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


import keras.backend as K


# In[2]:


#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# In[3]:


#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = .98)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# In[4]:


# Load the TensorBoard notebook extension.
get_ipython().run_line_magic('reload_ext', 'tensorboard')


# In[5]:


from numpy.random import seed
seed(1)


# In[6]:


os.chdir(os.path.dirname(os.path.abspath('Documentos/git-repos/doutorado/')))

def current_path(): 
    print("Current working directory ") 
    print(os.getcwd()) 
    

print(current_path())


# In[7]:


# Versões dos pacotes usados neste jupyter notebook
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Carlos Eduardo Veras Neves" --iversions')


# In[8]:




# # Carrega os dados

# In[9]:


# desembolsos = pd.read_csv('desembolsos.csv')
# pib = pd.read_csv('pib.csv')
# fbcf = pd.read_csv('fbcf.csv') 
github_repo = 'https://raw.githubusercontent.com/carlosvneves/doutorado/master/'
desembolsos = pd.read_csv(github_repo + 'desembolsos.csv')
pib = pd.read_csv(github_repo +'pib.csv')
fbcf = pd.read_csv(github_repo +'fbcf.csv') 


# In[10]:


fbcf.index = pd.to_datetime(fbcf['date'])
fbcf.drop(['date'],inplace=True, axis = 1)
fbcf.head()


# In[11]:


pib.index = pd.to_datetime(pib['date'])
pib.drop(['date'],inplace=True, axis = 1)
pib.head()


# In[12]:


desembolsos.head()


# In[13]:


desembolsos.index = pd.to_datetime(desembolsos['date'])
desembolsos.drop(['date'], inplace=True, axis = 1)
desembolsos.head()


# In[14]:


data = desembolsos.groupby(pd.PeriodIndex(desembolsos.index, freq='Q')).mean()
data = data.loc['1996Q1':]
data.index = data.index.to_timestamp(freq='Q')

for col in data.columns:
    data[col] = data[col]/pib['pib'].values * 100

data.head()


# In[15]:


# corte da série de acordo com a análise de tendência
start = '2002Q1'

data['Investimento'] = fbcf['fbcf'].values/pib['pib'].values *100
data = data.loc[start:]

data.describe()


# # Visualiza os dados

# In[16]:


data[['Investimento']].plot(figsize=(12,10));


# In[17]:


data[['Agropecuária','Indústria','Infraestrutura','Comércio e serviços', 'Total']].plot(figsize=(12,10));


# # Ajuste Sazonal (X-13 ARIMA) das séries para otimizar a modelagem

# In[18]:


X13_PATH = 'doutorado/x13/'

data_sa = pd.DataFrame(data)
data_sa.rename(columns=lambda x: x[0:3], inplace=True)

for col in data_sa:
    sa = sm.tsa.x13_arima_analysis(data_sa[col],x12path=X13_PATH)
    data_sa[col] = sa.seasadj.values

data_sa.tail()


# # Visualiza dados ajustados

# In[24]:


data_sa[['Agr','Ind','Inf','Com','Tot']].plot(figsize=(12,8));


# In[25]:


data_sa['Inv'].plot(figsize=(12,8));


# # Prepara dados para modelo autorregressivo

# In[26]:


data_lag1 = data_sa.shift(1).fillna(0)
data_lag1.head()


# In[27]:


data_lag2 = data_sa.shift(2).fillna(0)
data_lag2.head()


# In[28]:


data_lag3 = data_sa.shift(3).fillna(0)
data_lag3.head()


# In[29]:


df = pd.concat([data_sa,data_lag1['Inv'],data_lag2['Inv'],data_lag3['Inv']], axis=1, sort=False)
df.columns = ['Agr','Ind','Inf','Com','Tot','Inv','Inv-1','Inv-2','Inv-3']
df.head()


# In[30]:


df.corr()


# # Modelo de Redes Neurais Recorrentes (RNN)

# ### Conversão da estrutura de séries para aprendizado supervisionado

# In[31]:


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


# ### Mede performance do modelo

# In[32]:


# Função
from sklearn.metrics import mean_squared_error

def performance(y_true, y_pred): 
        mse = mean_squared_error(y_true,y_pred)
        mape = mean_squared_error(y_true,y_pred)
        print('MSE das previsões é {}'.format(round(mse, 2))+
                      '\nRMSE das previsões é {}'.format(round(np.sqrt(mse), 2))+
                      '\nMAPE das previsões é {}'.format(round(mape, 2)))
        return mse


# ### Treino da RNA

# In[33]:


def train_model(data, cfg):

      n_endog,n_quarters, n_train_quarters, n_features, n_nodes, n_epochs, n_batch = cfg

      values = data.values
      values = values.astype('float32')
      scaler = MinMaxScaler(feature_range=(0, 1))
      scaled = scaler.fit_transform(values)
      # specify the number of lag quarters
      # frame as supervised learning
      reframed = series_to_supervised(scaled, n_quarters, 1)

      # split into train and test sets
      values = reframed.values
      
      train = values[:n_train_quarters, :]
      test = values[n_train_quarters:, :]
      
      n_obs = n_quarters * n_features
      #n_obs = n_quarters * n_endog    
    
      #train = values[:n_quarters, :]
      #test = values[n_quarters:, :]   
      # split into input and outputs
      #n_obs = n_quarters * n_features
      #n_obs = n_quarters * n_endog  
      train_X, train_y = train[:, :n_obs], train[:, -n_features]
      test_X, test_y = test[:, :n_obs], test[:, -n_features]
      #train_X, train_y = train[:, :n_obs], train[:, -n_endog]
      #test_X, test_y = test[:, :n_obs], test[:, -n_endog]
      
      #print(train_X.shape[0])
      # reshape input to be 3D [samples, timesteps, features]
      train_X = train_X.reshape((train_X.shape[0], n_quarters, n_features))
      test_X = test_X.reshape((test_X.shape[0], n_quarters, n_features))
      #train_X = train_X.reshape((train_X.shape[0], n_quarters, n_endog))
      #test_X = test_X.reshape((test_X.shape[0], n_quarters, n_endog))  
      # Criação e treinamento do modelo LSTM Padrão
      #n_input = n_quarters*n_features
      #n_input = n_quarters*n_endog
      n_neurons = n_nodes
      batch_size=n_batch


      # reshape input to be 3D [samples, timesteps, features]
      #train_X = train_X.reshape((train_X.shape[0], n_quarters, n_features))
      #test_X = test_X.reshape((test_X.shape[0], n_quarters, n_features))
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


      model.add(keras.layers.LSTM(n_neurons,activation = 'tanh',recurrent_activation = 'sigmoid',
                        recurrent_dropout = 0,unroll = False, use_bias = True,
                        input_shape=(train_X.shape[1], train_X.shape[2])))
      model.add(keras.layers.Dropout(dropout))
      model.add(keras.layers.Dense(hidden_nodes, activation = 'relu'))
      model.add(keras.layers.Dropout(dropout))
      model.add(keras.layers.Dense(1))

      learning_rate=1.0e-3
      
      #session = K.get_session()
      #weights_initializer = tf.compat.v1.variables_initializer(layer.weights)
      #session.run(weights_initializer)  
      
      opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

      model.compile(loss="mse", optimizer=opt, metrics=["mae"])

      tensorboard = TensorBoard(log_dir="logs/{}-{}-{}-{}-".format(n_features, n_nodes, n_epochs, n_batch) + datetime.now().strftime("%Y%m%d-%H%M%S"))  
      patience_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

      # fit network
      model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, 
                validation_data=(test_X, test_y), verbose=0, shuffle=False, 
                callbacks=[patience_callback, tensorboard],use_multiprocessing=True,workers=128)
      
      model.save("model{}-{}-{}-{}--{}.h5".format(n_features, n_nodes, n_epochs, 
                                                   n_batch,datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))  
  
      return model, test_X, test_y,scaler


# ### Avaliação do Modelo

# In[34]:


def eval_model(data, cfg, n_rep=2):
  
      n_endog,n_quarters, n_train_quarters, n_features, n_nodes, n_epochs, n_batch = cfg

      #n_input = n_quarters*(n_features)
      #n_input = n_quarters*(n_endog+1)  


      #resultado = np.zeros((n_input,n_rep))
      resultado = []
      perf = np.zeros((n_rep))

      # Loop (TODO)
      # Vamos repetir o processo de treinamento por 20 vezes e armazenar todos os resultados, pois assim usaremos
      # diferentes amostras. Ao final, tiramos a média para encontrar as previsões. 
      # make a prediction
      print('\n')
      print('##'*35)
      series_par = "{}-n_endog,{}-n_quarters,{}-n_train_quarters,{}-n_features".format(n_endog,
                                                                                       n_quarters, 
                                                                                       n_train_quarters, 
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
        test_x = test_x.reshape((test_x.shape[0], n_quarters*n_features))
        #test_x = test_x.reshape((test_x.shape[0], n_quarters*n_endog))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_x[:, -n_endog:]), axis=1)##
        #inv_yhat = concatenate((yhat, test_x[:, -n_features:]), axis=1)###
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
      hiper = cfg
      resultado = np.array(resultado) 
      
      # Loop para gerar as previsões finais
      result_mean = np.zeros((resultado.shape[1],1))
      for i in range(resultado.shape[1]):
        result_mean[i] = np.mean(resultado[:,i])
      
      model.save("model{}-{}.h5".format(model_par,datetime.now().strftime("%Y%m%d-%H%M%S")))  
    
      #model.summary()
        
      return result_mean, perf_mean, cfg
  


# ### Grid Search

# In[35]:


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


# ### Seleciona o melhor resultado

# In[36]:


def best_result(error, result, hyperparms):
      index = error.index(min(error))
      best_fit = result[index]
      best_hyper = hyperparms[index]
      print(f"Parâmetros do melhor modelo: {best_hyper}")
      print(f"Menor MSE: {round(min(error),3)}")
      return best_fit


# ### Execução do Modelo

# In[37]:


import time

def run_model(data,cfg):
      start = time.time()


      results, error, hyperparams = grid_search(data, cfg)

      print(f"MSE: ({min(error)})\n")


      end = time.time()
      hours, rem = divmod(end-start, 3600)
      minutes, seconds = divmod(rem, 60)
      print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
  
      return results, error, hyperparams


# ### Configuração do modelo

# In[38]:


# Lista de hiperparâmetros que serão testados
def config_model(n_endog = [4], n_quarters = [8], n_train_quarters = [24], n_features = [5],
                 n_nodes = [150,300],n_epochs = [100],n_batch = [128] ):
# forma de inserção manual dos dados: [[5],[8],[24],[5],[150,300],[50,100],[72]]
    
    configs = list()
    for i in n_endog:
        for j in n_quarters:
            for k in n_train_quarters:
                for l in n_features:
                    for m in n_nodes:
                        for n in n_epochs:
                          for o in n_batch:
                            cfg = [i,j,k, l, m, n, o]
                            configs.append(cfg)
                              
    print('\nTotal de Combinações de Hiperparâmetros: %d' % len(configs))
    return configs


# # Simulação

# In[39]:


var = ['Inv', 'Agr', 'Ind', 'Inf', 'Com']
var_ar = ['Inv', 'Agr', 'Ind', 'Inf', 'Com','Inv-1']


# ## Modelo LSTM - sem termos Autorregressivos

# In[40]:


config = config_model(n_quarters = [8,12],n_train_quarters = [24,36],n_batch = [16,32])
results_lstm, error_lstm, hyperparams_lstm = run_model(data_sa[var], config)


# In[36]:


resultado_lstm = best_result(error_lstm, results_lstm, hyperparams_lstm)


# ## Modelo LSTM-com termos Autorregressivos

# ### Configuração do Modelo

# In[37]:


config = config_model(n_endog=[5], n_features=[6],n_quarters = [8,12],n_train_quarters = [24,36],n_batch = [16,32]) # como o modelo incorpora 'Inv-1' ==> n_endog=[5]
results_lstmar, error_lstmar, hyperparams_lstmar = run_model(df[var_ar], config)


# In[43]:


resultado_lstmar = best_result(error_lstmar, results_lstmar, hyperparams_lstmar)


# # Visualiza a previsão do modelo com menor MSE

# In[44]:


# Plot
plt.figure(figsize = (20, 6))

# Série original
plt.plot(data_sa.index, 
          data_sa['Inv'].values,
          label = 'Valores Observados',
          color = 'Red')

# Previsões
plt.plot(data_sa.index[-len(resultado_lstmar):], 
         resultado_lstmar,
         label = 'Previsões com Modelo de Redes Neurais LSTM-AR', 
         color = 'Blue')


plt.plot(data_sa.index[-len(resultado_lstm):], 
         resultado_lstm,
         label = 'Previsões com Modelo de Redes Neurais LSTM', 
         color = 'Black')


plt.title('Previsões com Modelo de Redes Neurais Recorrentes')
plt.xlabel('Ano')
plt.ylabel('Investimento (%PIB)')
plt.legend()
plt.show()


# # Previsões

# ## Carrega os modelos com melhor desempenho

# In[60]:


from keras.models import load_model
model_lstm = keras.models.load_model('c:/temp/model5-150-100-32--2020-06-26-14-02-33.h5')
model_lstm.summary()


# In[61]:


data_sa.tail()


# In[62]:


matrix = np.zeros([8,5])
df_fcast = pd.DataFrame(matrix)
df_fcast.index = pd.date_range(start='2020-01-31',end='2022-01-31', freq='Q')
df_fcast.columns = ['Agr','Ind','Inf','Com','Inv']
df_fcast


# In[76]:


n_endog,n_quarters, n_train_quarters, n_features, n_nodes, n_epochs, n_batch = [4, 8, 24, 5, 150, 100, 32]
n_rep=1
#n_input = n_quarters*n_features
n_input = n_quarters*n_endog 

modelo = model_lstm
resultado = np.zeros((n_input,n_rep))
perf = np.zeros((n_rep))

print(df_fcast.shape)
values = values = df_fcast.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag quarters
# frame as supervised learning
reframed = series_to_supervised(scaled, 0, n_features)
print(reframed)
print(reframed.shape)


input_x = array(reframed)
input_x = input_x.reshape((1, df_fcast.shape[0], n_features))
#print(input_x.shape)
#print(scaled)
# split into train and test sets
#values = arrat#reframed.values
# #n_obs = n_train_quarters * n_features
# train = values[:n_train_quarters, :]
#test = array(df_fcast)#values[:, :]
#print(test.shape)
# # split into input and outputs
# n_obs = n_quarters * n_features
# train_X, train_y = train[:, :n_obs], train[:, -n_features]
# test_X, test_y = test[:, :n_obs], test[:, -n_features]

# # reshape input to be 3D [samples, timesteps, features]
# train_X = train_X.reshape((train_X.shape[0], n_quarters, n_features))
#test_X = test.reshape((1, test.shape[0], test.shape[1]))
#print(test_X.shape)
modelo.predict(input_x)



#       # Loop (TODO)
#       # Vamos repetir o processo de treinamento por 20 vezes e armazenar todos os resultados, pois assim usaremos
#       # diferentes amostras. Ao final, tiramos a média para encontrar as previsões. 
#       # make a prediction
#       print('\n')
#       print('##'*35)
#       series_par = "{}-n_endog,{}-n_quarters,{}-n_train_quarters,{}-n_features".format(n_endog,
#                                                                                        n_quarters, 
#                                                                                        n_train_quarters, 
#                                                                                        n_features)
      
#       model_par =  "{}-n_nodes,{}-n_epochs,{}-n_batch".format(n_nodes, n_epochs, n_batch)                                                                                                          
                                                                                                                  
      
#       print(f'## Avaliação do Modelo: \n{series_par}\n{model_par}\n ## ')
#       print(datetime.now().strftime("%Y/%m/%d-%H:%M:%S\n"))

#x = values.reshape((1,1,300))
#for i in range(n_rep):

    #modelo, testx, testy, scaler = train_model(data, cfg)
    #modelo.model.model
    #test_x = testx
    #test_y = testy

    #yhat = modelo.predict(test_x)
    #test_x = test_x.reshape((test_x.shape[0], n_quarters*n_features))
    # invert scaling for forecast
    #inv_yhat = concatenate((yhat, test_x[:, -n_endog:]), axis=1)#concatenate((yhat, test_x[:, -n_features:]), axis=1)#
    #yhat = scaler.inverse_transform(inv_yhat)
    #yhat = yhat[:,0]


    #resultado[:,i] = yhat

    #print(f'\nRepetição:{i+1}')
    #print(f'# épocas:({n_epochs}) # neurônios:({n_nodes}) # batch:({n_batch})')
    #print(f'loss:{round(modelo.history.history["loss"][-1],4)} - end val_loss: {round(modelo.history.history["val_loss"][-1],4)}\n')
#print(yhat)

    # invert scaling for actual
    #test_y = test_y.reshape((len(test_y), 1))
    #inv_y = concatenate((test_y, test_x[:, -n_endog:]), axis=1)
    #y = scaler.inverse_transform(inv_y)
    #y = y[:,0]
    
#         perf[i] = performance(y,yhat)

#       perf_mean = np.mean(perf)
#       hiper = cfg

      # Loop para gerar as previsões finais
#       result_mean = np.zeros((resultado.shape[0],1))
#       for i in range(resultado.shape[0]):
#         result_mean[i] = np.mean(resultado[i,:])


# In[200]:


def forecast(data, modelo, cfg, n_rep=1):
  
      n_endog,n_quarters, n_train_quarters, n_features, n_nodes, n_epochs, n_batch = cfg
      
        
      print(data.values)  
      values = data.values
      values = values.astype('float32')
      scaler = MinMaxScaler(feature_range=(0, 1))
      scaled = scaler.fit_transform(values)
      # specify the number of lag quarters
      # frame as supervised learning
      reframed = series_to_supervised(scaled, n_quarters, 1)
        
      # split into train and test sets
      values = reframed.values
      #n_obs = n_train_quarters * n_features
      #train = values[:n_train_quarters, :]
      test = values[:, :]  
      # split into input and outputs
      n_obs = n_quarters * n_features
      #train_X, train_y = train[:, :n_obs], train[:, -n_features]
      test_X = test[:, :]

      # reshape input to be 3D [samples, timesteps, features]
      #train_X = train_X.reshape((train_X.shape[0], n_quarters, n_features))
      test_X = test_X.reshape((test_X.shape[0], n_quarters, n_features))

      # Criação e treinamento do modelo LSTM Padrão
      #n_input = n_quarters*n_features
      n_neurons = n_nodes
      batch_size=n_batch
    
    
    
      #n_input = n_quarters*n_features
      n_input = n_quarters*n_endog  


      resultado = np.zeros((n_input,n_rep))
      perf = np.zeros((n_rep))

      # Loop (TODO)
      # Vamos repetir o processo de treinamento por 20 vezes e armazenar todos os resultados, pois assim usaremos
      # diferentes amostras. Ao final, tiramos a média para encontrar as previsões. 
      # make a prediction
      print('\n')
      print('##'*35)
#       series_par = "{}-n_endog,{}-n_quarters,{}-n_train_quarters,{}-n_features".format(n_endog,
#                                                                                        n_quarters, 
#                                                                                        n_train_quarters, 
#                                                                                        n_features)

      #model_par =  "{}-n_nodes,{}-n_epochs,{}-n_batch".format(n_nodes, n_epochs, n_batch)                                                                                                          


      #print(f'## Avaliação do Modelo: \n{series_par}\n{model_par}\n ## ')
      #print(datetime.now().strftime("%Y/%m/%d-%H:%M:%S\n"))

      for i in range(n_rep):

        #modelo, testx, testy, scaler = train_model(data, cfg)
        #modelo.model.model
        #test_x = testx
        #test_y = testy
        print(test_X)
        yhat = modelo.predict(test_X, verbose=2)
        #test_x = test_x.reshape((test_x.shape[0], n_quarters*n_features))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, -n_endog:]), axis=1)#concatenate((yhat, test_x[:, -n_features:]), axis=1)#
        yhat = scaler.inverse_transform(inv_yhat)
        yhat = yhat[:,0]


        resultado[:,i] = yhat

        #print(f'\nRepetição:{i+1}')
        #print(f'# épocas:({n_epochs}) # neurônios:({n_nodes}) # batch:({n_batch})')
        #print(f'loss:{round(modelo.history.history["loss"][-1],4)} - end val_loss: {round(modelo.history.history["val_loss"][-1],4)}\n')


        # invert scaling for actual
        #test_y = test_y.reshape((len(test_y), 1))
        #inv_y = concatenate((test_y, test_x[:, -n_endog:]), axis=1)
        #y = scaler.inverse_transform(inv_y)
        #y = y[:,0]

        #perf[i] = performance(y,yhat)

      #perf_mean = np.mean(perf)
      #hiper = cfg

      # Loop para gerar as previsões finais
      result_mean = np.zeros((resultado.shape[0],1))
      for i in range(resultado.shape[0]):
        result_mean[i] = np.mean(resultado[i,:])

      #modelo.save("model{}-{}.h5".format(model_par,datetime.now().strftime("%Y%m%d-%H%M%S")))  

      return result_mean
  


# In[184]:


from keras.models import Model, Sequential
from keras import backend as K

def create_dropout_predict_function(model, dropout):
    """
    Create a keras function to predict with dropout
    model : keras model
    dropout : fraction dropout to apply to all layers
    
    Returns
    predict_with_dropout : keras function for predicting with dropout
    """
    
    # Load the config of the original model
    conf = model.get_config()
    # Add the specified dropout to all layers
    for layer in conf['layers']:
        # Dropout layers
        if layer["class_name"]=="Dropout":
            layer["config"]["rate"] = dropout
        # Recurrent layers with dropout
        elif "dropout" in layer["config"].keys():
            layer["config"]["dropout"] = dropout

    # Create a new model with specified dropout
    if type(model)==Sequential:
        # Sequential
        model_dropout = Sequential.from_config(conf)
    else:
        # Functional
        model_dropout = Model.from_config(conf)
    model_dropout.set_weights(model.get_weights()) 
    
    # Create a function to predict with the dropout on
    predict_with_dropout = K.function(model_dropout.inputs+[K.learning_phase()], model_dropout.outputs)
    
    return predict_with_dropout


# In[187]:


data_sa.tail()


# In[198]:


matrix = np.zeros([8,4])
df_fcast = pd.DataFrame(matrix)
df_fcast.index = pd.date_range(start='2020-01-31',end='2022-01-31', freq='Q')
df_fcast.columns = ['Agr','Ind','Inf','Com']
df_fcast


# In[199]:


series_to_supervised(df_fcast)


# In[209]:


modelo = model_lstm
data = df_fcast
#forecast(df_fcast, model_lstm,cfg)

n_endog,n_quarters, n_train_quarters, n_features, n_nodes, n_epochs, n_batch = cfg = [4, 8, 24, 5, 300, 100, 128]


#print(data.values) 

values = data.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

print(scaled) 


# In[217]:


# specify the number of lag quarters
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# split into train and test sets
values = reframed.values
#n_obs = n_train_quarters * n_features
#train = values[:n_train_quarters, :]
test = values[:, :]  
# split into input and outputs
n_obs = n_quarters * n_features
#train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X = test[:, :]
print(reframed)


# In[223]:



# reshape input to be 3D [samples, timesteps, features]
#train_X = train_X.reshape((train_X.shape[0], n_quarters, n_features))
test_X = test_X.reshape((test_X.shape[0], 2, 4))

print(test_X)


# In[225]:


# Criação e treinamento do modelo LSTM Padrão
#n_input = n_quarters*n_features
n_neurons = n_nodes
batch_size=n_batch



#n_input = n_quarters*n_features
n_input = n_quarters*n_endog  


#resultado = np.zeros((n_input,n_rep))
#perf = np.zeros((n_rep))

# Loop (TODO)
# Vamos repetir o processo de treinamento por 20 vezes e armazenar todos os resultados, pois assim usaremos
# diferentes amostras. Ao final, tiramos a média para encontrar as previsões. 
# make a prediction
print('\n')
print('##'*35)
#       series_par = "{}-n_endog,{}-n_quarters,{}-n_train_quarters,{}-n_features".format(n_endog,
#                                                                                        n_quarters, 
#                                                                                        n_train_quarters, 
#                                                                                        n_features)

#model_par =  "{}-n_nodes,{}-n_epochs,{}-n_batch".format(n_nodes, n_epochs, n_batch)                                                                                                          


#print(f'## Avaliação do Modelo: \n{series_par}\n{model_par}\n ## ')
#print(datetime.now().strftime("%Y/%m/%d-%H:%M:%S\n"))

#for i in range(n_rep):

#modelo, testx, testy, scaler = train_model(data, cfg)
#modelo.model.model
#test_x = testx
#test_y = testy
print(test_X)
yhat = modelo.predict(test_X, verbose=2)

print(yhat)

#test_x = test_x.reshape((test_x.shape[0], n_quarters*n_features))
# invert scaling for forecast
#inv_yhat = concatenate((yhat, test_X[:, -n_endog:]), axis=1)#concatenate((yhat, test_x[:, -n_features:]), axis=1)#
#yhat = scaler.inverse_transform(inv_yhat)
#yhat = yhat[:,0]


#resultado[:,i] = yhat

#print(f'\nRepetição:{i+1}')
#print(f'# épocas:({n_epochs}) # neurônios:({n_nodes}) # batch:({n_batch})')
#print(f'loss:{round(modelo.history.history["loss"][-1],4)} - end val_loss: {round(modelo.history.history["val_loss"][-1],4)}\n')


# invert scaling for actual
#test_y = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_y, test_x[:, -n_endog:]), axis=1)
#y = scaler.inverse_transform(inv_y)
#y = y[:,0]

#perf[i] = performance(y,yhat)

#perf_mean = np.mean(perf)
#hiper = cfg

# Loop para gerar as previsões finais
#result_mean = np.zeros((resultado.shape[0],1))
#for i in range(resultado.shape[0]):
#result_mean[i] = np.mean(resultado[i,:])


# In[50]:


import numpy as np

dropout = 0.5
num_iter = 20
num_samples = input_data[0].shape[0]

#path_to_model = "c:/temp/.hdf5"
#model = load_model(path_to_model)

predict_with_dropout = create_dropout_predict_function(model, dropout)

predictions = np.zeros((num_samples, num_iter))
for i in range(num_iter):
    predictions[:,i] = predict_with_dropout(input_data+[1])[0].reshape(-1)


# In[51]:


ci = 0.8
lower_lim = np.quantile(predictions, 0.5-ci/2, axis=1)
upper_lim = np.quantile(predictions, 0.5+ci/2, axis=1)

