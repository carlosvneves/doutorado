#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:55:20 2020

@author: nnlinux
"""


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
