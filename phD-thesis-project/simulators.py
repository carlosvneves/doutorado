#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:42:04 2020

@author: root
"""
from lib import * 


TENSOR_BOARD_LOG = False

class Simulator:

	

    #%%Treino da RNA
    def train_model(self, cfg):
    
      n_endog,n_steps, n_train_steps, n_features, n_nodes, n_epochs, n_batch = cfg
    
      values = self.data.values
      values = values.astype('float32')
      scaler = MinMaxScaler(feature_range=(0, 1))
      scaled = scaler.fit_transform(values)
      # specify the number of lag quarters
      # frame as supervised learning
      reframed = self.series_to_supervised(scaled, n_steps, 1)
    
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
    def eval_model(self, cfg, n_rep=10):
      
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
          
          model, testx, testy, scaler = self.train_model(cfg)
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
          
          perf[i] = self.performance(y,yhat)
          
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
    def grid_search(self):
            errors = []
            results = []
            hyperparams = []
            #resultado = avalia_modelo(modelo, test_X, config[0])
            # Gera os scores
            config = self.config
            for cfg in config:
              result,error,hyper = self.eval_model(cfg) 
              results.append(result)
              errors.append(error)
              hyperparams.append(hyper)
    
            # Ordena os hiperparâmetros pelo erro
            #errors.sort(key = lambda tup: tup[1])
            return results,errors, hyperparams
    
    
    #%%Seleciona o melhor resultado
    def best_result(self, error, result, hyperparms):
          index = error.index(min(error))
          best_fit = result[index]
          best_hyper = hyperparms[index]
          print(f"Parâmetros do melhor modelo:{best_hyper}")
               
          print(f"Menor MSE: {round(min(error),3)}")
          return best_fit, best_hyper
    
    
    #%% Execução do Modelo com grid searh
    def run_grid_search(self):
          start = time.time()
    
    
          results, error, hyperparams = self.grid_search()
    
          print(f"MSE: ({min(error)})\n")
    
    
          end = time.time()
          hours, rem = divmod(end-start, 3600)
          minutes, seconds = divmod(rem, 60)
          print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
      
          return results, error, hyperparams

    #%% Conversão da estrutura de séries para aprendizado supervisionado
    # convert series to supervised learning
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
            n_vars = 1 if type(data) is list else self.data.shape[1]
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
    def performance(self,y_true, y_pred): 
            mse = mean_squared_error(y_true,y_pred)
            mape = mean_squared_error(y_true,y_pred)
            print('MSE das previsões é {}'.format(round(mse, 2))+
                          '\nRMSE das previsões é {}'.format(round(np.sqrt(mse), 2))+
                          '\nMAPE das previsões é {}'.format(round(mape, 2)))
            return mse

    #%% Realiza previsão fora da amostra 
    def forecast(self, n_inputs, n_predictions, model):
            
    # =============================================================================
    #     model = load_model('model-4-n_endog,36-n_quarters,36-n_train_steps,5-n_features-300-n_nodes,100-n_epochs,32-n_batch.h5')
    #     print(model.summary())
    #     df = load_data()    
    #     df = df[['Inv', 'Agr', 'Ind', 'Inf', 'Com']] 
    # =============================================================================
      
        
        n_exog = self.data.shape[1]-1
        n_features = self.data.shape[1]
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
            
        values = self.data.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        
        #n_quarters = 36
        
        n_features = self.data.shape[1]
        
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
        
        add_dates = [self.data.index[-1] + pd.DateOffset(months=x) for x in range(0,n_steps+1) ]
        future_dates = pd.DataFrame(index=add_dates[1:],columns=self.data.columns)
        
        df_predict = pd.DataFrame(yhat,
                              index=future_dates[-n_steps:].index, columns=['Prediction'])
        
        df_proj = pd.concat([self.data,df_predict], axis=1)
       
        
        return df_proj, pred_list

    #%% Função que simula modelo LSTM
    def LSTM(self):
        from numpy.random import seed
        seed(1)
           
        # Load the TensorBoard notebook extension.
        get_ipython().run_line_magic('reload_ext', 'tensorboard')
        
        df = self.data
            
        var = ['Inv', 'Agr', 'Ind', 'Inf', 'Com']
        var_ar = ['Inv', 'Agr', 'Ind', 'Inf', 'Com','Inv-1']
    
    
        #%% Modelo LSTM - sem termos Autorregressivos %
        #config = config_model(n_steps = [24,36],n_train_steps = [24,36],n_batch = [16,32])
        results_lstm, error_lstm, hyperparams_lstm = self.run_grid_search()
        # armazena o melhor resultado   
        best_lstm_res, best_lstm_par = self.best_result(error_lstm, results_lstm, hyperparams_lstm)
        
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
        
        return best_lstm_res, best_lstm_par
    
    
    #%% Setters
    def set_n_endog(self, n_endog):
        self.n_endog = list(n_endog)
        
    def set_n_steps(self, n_steps):
        self.n_steps = list(n_steps)
        
    def set_n_train_steps(self, n_train_steps):
        self.n_train_steps = list(n_train_steps)
        
    def set_n_features(self, n_features):
        self.n_features= list(n_features)
        
    def set_n_nodes(self, n_nodes):
        self.n_nodes = list(n_nodes)
        
    def set_n_epochs(self, n_epochs):
        self.n_epochs = list(n_epochs)
        
    def set_n_batch(self, n_batch):
        self.n_batch = list(n_batch)
        
    def set_data(self, data):
        self.data = data

    #%% Carrega dados do melhor modelo
    def load_best_model():
        
        with open ('best_lstm_par.pkl', 'rb') as fp:
            best_par = pickle.load(fp)
       
        print(best_par)
        
                
        best_file = 'model-pct-{}-n_endog,{}-n_steps,{}-n_train_steps,{}-n_features-{}-n_nodes,{}-n_epochs,{}-n_batch.h5'.format(best_par[0],
                                                                                                                                 best_par[1],
                                                                                                                                 best_par[2],
                                                                                                                                 best_par[3],
                                                                                                                                 best_par[4],
                                                                                                                                 best_par[5],
                                                                                                                                 best_par[6])
        
        #=========================================================================
        best_model = keras.models.load_model(best_file)    
        
        print('##'*25)
        print('# Melhor Modelo :')
        best_model.summary()
        print('##'*25)
        
        with open ('best_lstm_res.pkl', 'rb') as fp:
            best_res = pickle.load(fp)
    
                 
        return best_model, best_res, best_par[1]
    
    #%% Class constructor
    def __init__(self, data, config):
            
            self.config = config
            self.n_endog = config[0]
            self.n_steps= config[1] 
            self.n_train_steps = config[2] 
            self.n_features = config[3] 
            self.n_nodes = config[4] 
            self.n_epochs = config[5] 
            self.n_batch = config[6]
            self.data = data


def main():
    print('Simulação do Modelo')
   
    df = load_data()
    var = ['Inv', 'Agr', 'Ind', 'Inf', 'Com']

    config = config_model(n_steps = [24,36],n_train_steps = [24,36],n_batch = [16,32])
    simulator = Simulator(df[var], config)
    simulator.LSTM()
    
if __name__ == '__main__':
    main()
    

            
    


    #%% Modelo LSTM - sem termos Autorregressivos %
    #results_lstm, error_lstm, hyperparams_lstm = run_model(df[var], config)