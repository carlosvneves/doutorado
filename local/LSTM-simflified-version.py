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


os.chdir(os.path.dirname(os.path.abspath('nnlinux/Documentos/git-repos/doutorado')))

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




# In[18]:
### Ajuste Sazonal (X-13 ARIMA) das séries para otimizar a modelagem

X13_PATH = 'doutorado/x13/'

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


# In[30]:


print(df.corr())
print(df.describe())

"""
Prepare de dataset for NN trainning and evaluate

Input -> Predict

Quarter i -> Quarter i+1
Quarter i+1 -> Quarter i+2

"""

df_train = df.loc[:'2016-12-31',:]
df_test = df.loc['2017-03-31':,:]

# train-test split
print(df_train.tail())
print(df_test.head())


print(df_train.shape, df_test.shape)

# the endogenous variable is the 0-index
data_train = df_train[['Inv','Agr','Ind','Inf','Com']]
data_train.head()

data_train = np.array(data_train)
X_train, y_train = [], []

# Prepare sequence for n_qtr - number of past time steps to make a prediction
n_qtr = 4
# number of predictors in a multivariate dataset (exogenous variables)
n_features = 4

for i in range(n_qtr, len(data_train)-n_qtr):
    X_train.append(data_train[i-n_qtr:i,1:n_features+1])
    y_train.append(data_train[i:i+n_qtr,0])
    
    
X_train, y_train = np.array(X_train), np.array(y_train)

# Normalize data
x_scaler = MinMaxScaler()

for i in range(0,len(X_train)):
    X_train[i] = x_scaler.fit_transform(X_train[i,:,:])

y_scaler = MinMaxScaler()
y_train  = y_scaler.fit_transform(y_train)

### Build LSTM

reg = Sequential()
reg.add(LSTM(units = 200, activation = 'relu', input_shape=(n_qtr,n_features),
             return_sequences=False))
reg.add(Dropout(0.3))
reg.add(Dense(4))

opt = tf.keras.optimizers.RMSprop(learning_rate=1.0e-4)

reg.compile(loss='mse', optimizer=opt)
history = reg.fit(X_train,y_train, epochs=200,batch_size=12,
                  validation_split=0.1,shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

###########################################################
### Evaluate Model: Prepare test dataset and test the LSTM
data_test= df_test[['Inv','Agr','Ind','Inf','Com']]
data_test.head()

data_test = np.array(data_test)
X_test, y_test = [], []


for i in range(n_qtr, len(data_test)-n_qtr):

    X_test.append(data_test[i-n_qtr:i,1:n_features+1])
    y_test.append(data_test[i:i+n_qtr,0])
    
    
X_test, y_test = np.array(X_test), np.array(y_test)

# Normalize data
for i in range(0,len(X_test)):
    X_test[i] = x_scaler.fit_transform(X_test[i,:,:])


y_test  = y_scaler.fit_transform(y_test)

# Reshape test data to evaluate NN
#X_test = X_test.reshape(1,10,4)

# Make prediction
yhat = reg.predict(X_test) 


yhat = y_scaler.inverse_transform(yhat)
y_true = y_scaler.inverse_transform(y_test)

def sequence_to_series(sequence,n_steps=4):
    a = np.zeros(n_steps)
    a[0] = 1
    b = np.zeros(n_steps)
    b[n_steps-1] = 1
    a = a.dot(sequence)
    b = b.dot(sequence)
    b = b[1:n_steps]

    series = np.concatenate((a,b),axis=None)
    
    return series 

yhat = sequence_to_series(yhat)
y_true = sequence_to_series(y_true)

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

print(evaluate_model(y_true,yhat))

print(np.std(y_true))

result = pd.concat([pd.Series(yhat),pd.Series(y_true)], axis = 1, sort=False)
result.index = df_test.index[n_qtr:len(df_test)-1]
result.columns = ['yhat','y']

print(result)

##############################################
# Make forecasts
agro = pd.Series(np.zeros(12))
ind = pd.Series(np.zeros(12))
inf = pd.Series(np.zeros(12))
com = pd.Series(np.zeros(12))

dates_forecast = pd.date_range(start='2020-01-01', periods=12, freq='Q')
df_forecast= pd.concat([agro, ind, inf, com], axis=1)
df_forecast.index = pd.DatetimeIndex(dates_forecast)

df_forecast = np.array(df_forecast)

X_forecast = []

for i in range(n_qtr, len(df_forecast)-n_qtr):

    X_forecast.append(df_forecast[i-n_qtr:i,0:n_features])

X_forecast = np.array(X_forecast)


ypred = reg.predict(X_forecast)

ypred = y_scaler.inverse_transform()

ypred = sequence_to_series(ypred)
