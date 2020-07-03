#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:16:28 2020

@author: nnlinux
"""


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
from keras.layers import LSTM, GRU#,CuDNNLSTM
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
from sklearn.preprocessing import MinMaxScaler, RobustScaler
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


#os.chdir(os.path.dirname(os.path.abspath('nnlinux/Documentos/git-repos/doutorado')))

#def current_path(): 
# =============================================================================
#     print("Current working directory ") 
#     print(os.getcwd()) 
# =============================================================================
    

#print(current_path())


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
#start = '2002Q1'

data['Investimento'] = fbcf['fbcf'].values/pib['pib'].values *100
#data = data.loc[start:]

data.describe()


# # Visualiza os dados

# In[16]:


data[['Investimento']].plot(figsize=(12,10));


# In[17]:


data[['Agropecuária','Indústria','Infraestrutura',
      'Comércio e serviços', 'Total']].plot(figsize=(12,10));




# In[18]:
### Ajuste Sazonal (X-13 ARIMA) das séries para otimizar a modelagem

X13_PATH = 'x13/'

data_sa = pd.DataFrame(data)
data_sa.rename(columns=lambda x: x[0:3], inplace=True)

for col in data_sa:
    sa = sm.tsa.x13_arima_analysis(data_sa[col],x12path=X13_PATH)
    data_sa[col] = sa.seasadj.values

data_sa.tail()



# In[24]:
# # Visualiza dados ajustados


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
print(df.head())
print(df.tail(24))


# Unsample data 
upsampled = df.resample('M')
#interpolated = upsampled.interpolate(method='spline', order=2)
interpolated = upsampled.interpolate(method='linear')
interpolated.tail(24)

df = interpolated
print(df.shape)

# In[30]:


print(df.corr())
print(df.describe())

"""
Prepare de dataset for NN trainning and evaluate

Input -> Predict

Quarter i -> Quarter i+1
Quarter i+1 -> Quarter i+2

"""

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps),:]
        Xs.append(v)
        ys.append(y[i + time_steps,0])
    return np.array(Xs), np.array(ys)

# Prepare sequence for n_qtr - number of past time steps to make a prediction
n_time_steps = 36


split = len(df)-n_time_steps
df_train = df.iloc[0:split,:]
df_test = df.iloc[-n_time_steps:,:]

######################## train-test split ###############################
print(df_train.tail())
print(df_test.head())


print(df_train.shape, df_test.shape)

############## train data
# the endogenous variable is the 0-index
data_train = df_train[['Inv','Agr','Ind','Inf','Com','Inv-1']]
data_train.head()
y_train = data_train['Inv']
X_train = data_train[['Agr','Ind','Inf','Com','Inv-1']]

# number of predictors in a multivariate dataset (exogenous variables)
n_features = X_train.shape[1]

X_train,y_train = np.array(X_train), np.array(y_train)

# Normalize data
x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(X_train)
#X_train = x_scaler.transform(X_train)


y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
#y_train = y_scaler.transform(y_train)

X_train, y_train = create_dataset(X_train, y_train,n_time_steps)
X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape, y_train.shape)

############## test data
# =============================================================================
# data_val= df_test[['Inv','Agr','Ind','Inf','Com']]
# print(data_val.head())
# 
# y_val = data_val['Inv']
# X_val = data_val[['Agr','Ind','Inf','Com']]
# 
# X_val,y_val = np.array(X_val), np.array(y_val)
# 
# # Normalize data
# X_val = x_scaler.transform(X_val)
# 
# y_val = y_scaler.transform(y_val.reshape(-1,1))
# 
# X_val, y_val = create_dataset(X_val, y_val,n_time_steps)
# X_val, y_val = np.array(X_val), np.array(y_val)
# XValidation, YValidation = X_val, y_val
# print(X_val.shape, y_val.shape)
# =============================================================================

### Build LSTM
reg = Sequential()
#reg.add(LSTM(units = 200, activation = 'relu', input_shape=(n_time_steps, n_features),
#             return_sequences=False))
reg.add(GRU(units = 200, activation = 'relu', input_shape=(n_time_steps, n_features),
             return_sequences=False))

#reg.add(Dense(units = 150, activation='relu'))
reg.add(Dropout(0.15))
reg.add(Dense(1))



opt = tf.keras.optimizers.RMSprop(learning_rate=1.0e-4)

batch = 24
n_epochs = 100


reg.compile(loss='mse', optimizer=opt)

#reg.save_weights('model.h5')

history = reg.fit(X_train,y_train, 
                  validation_split = 0.3,
                  epochs=n_epochs,batch_size=batch,shuffle=False)

#reg.load_weights('model.h5')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()


###########################################################
### Evaluate Model: Prepare test dataset and test the LSTM
data_test= df_test[['Inv','Agr','Ind','Inf','Com','Inv-1']]

y_test = data_test['Inv']
X_test = data_test[['Agr','Ind','Inf','Com','Inv-1']]

X_test,y_test = np.array(X_test), np.array(y_test)

# Normalize data
X_test = x_scaler.fit_transform(X_test)
#X_test = x_scaler.transform(X_test)

y_test = y_scaler.fit_transform(y_test.reshape(-1,1))
#y_test = y_scaler.transform(y_test.reshape(-1,1))

nqtr = 24

X_test, y_test  = create_dataset(X_test, y_test,nqtr)
X_test, y_test = np.array(X_test), np.array(y_test)

#

# Make prediction
yhat = reg.predict(X_test) 

print(yhat)

y_pred = y_scaler.inverse_transform(yhat.reshape(-1,1))
y_true = y_scaler.inverse_transform(y_test.reshape(-1,1))

print(y_true)
print(y_pred)



def evaluate_model(y_true, y_predicted):
    scores = []
    
    #calculate scores for each year
    for row in range(y_true.shape[0]):
        mse = (y_true[row] - y_predicted[row])**2
        rmse = np.sqrt(mse)
        scores.append(rmse)
    
    #calculate for the whole prediction
    total_score = 0
    for row in range(y_true.shape[0]):
        total_score = total_score + (y_true[row] - y_predicted[row])**2
    total_score = np.sqrt(total_score/(y_true.shape[0]*y_predicted.shape[0]))
    
    return total_score, scores

print(evaluate_model(y_true,y_pred))

print(np.std(y_true))


res = concatenate((y_pred,y_true), axis=1)
result = pd.DataFrame(res)
result.index = df_test.index[nqtr:len(df_test)]
result.columns = ['y_pred','y_true']

print(result)



##############################################
# Make forecasts
n_ahead = 12
n_before = 36
# =============================================================================
# 
# agro = pd.Series(np.zeros(n_ahead))
# ind = pd.Series(np.zeros(n_ahead))
# inf = pd.Series(np.zeros(n_ahead))
# com = pd.Series(np.zeros(n_ahead))
# inv = pd.Series(np.zeros(n_ahead))
# =============================================================================

# =============================================================================
# inv = pd.Series(np.zeros(n_ahead))
# agro = pd.Series(np.full(n_ahead,np.std(df_test['Agr'])))
# ind = pd.Series(np.full(n_ahead,np.std(df_test['Ind'])))
# inf = pd.Series(np.full(n_ahead,np.std(df_test['Inf'])))
# com = pd.Series(np.full(n_ahead,np.std(df_test['Com'])))
# =============================================================================


inv = pd.Series(np.zeros(n_ahead))
agro = pd.Series(np.full(n_ahead,np.mean(df.iloc[-n_before:]['Agr'])))
ind = pd.Series(np.full(n_ahead,np.mean(df.iloc[-n_before:]['Ind'])))
inf = pd.Series(np.full(n_ahead,np.mean(df.iloc[-n_before:]['Inf'])))
com = pd.Series(np.full(n_ahead,np.mean(df.iloc[-n_before:]['Com'])))
inv_lag1 = pd.Series(np.full(n_ahead,df.iloc[-n_ahead:]['Inv-1']))


df_forecast= pd.concat([inv,agro, ind, inf, com,inv_lag1], axis=1)
dates_forecast = pd.date_range(start='2020-01-01', periods=n_ahead, freq='M')
df_forecast.index = pd.DatetimeIndex(dates_forecast)
df_forecast.columns = data_train.columns

df_forecast = pd.concat((df.iloc[-n_before:][data_train.columns],df_forecast), 
                        axis=0)

y_forecast = df_forecast['Inv']
X_forecast = df_forecast[['Agr','Ind','Inf','Com','Inv-1']]

X_forecast,y_forecast = np.array(X_forecast), np.array(y_forecast)

# Normalize data
X_forecast = x_scaler.fit_transform(X_forecast)
#X_forecast = x_scaler.transform(X_forecast)

y_forecast = y_scaler.fit_transform(y_forecast.reshape(-1,1))
#y_test = y_scaler.transform(y_test.reshape(-1,1))


X_forecast, y_forecast  = create_dataset(X_forecast, y_forecast,n_before)
X_forecast, y_forecast = np.array(X_forecast), np.array(y_forecast)

#

# Make prediction
yhat = reg.fit(X_forecast)

print(yhat)

y_pred = y_scaler.inverse_transform(yhat.reshape(-1,1))
y_true = y_scaler.inverse_transform(y_forecast.reshape(-1,1))

print(y_true)
print(y_pred)
