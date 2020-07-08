#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:42:04 2020

@author: Carlos Eduardo Veras Neves - PhD candidate
University of Brasilia - UnB
Department of Economics - Applied Economics

Thesis Supervisor: Geovana Lorena Bertussi, PhD.
Title:Professor of Department of Economics - UnB
    
"""
from lib import * 
from models import *


#%% Parâmetros para utilização da GPU
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 1., allow_growth=True)
   
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)   

# Load the TensorBoard notebook extension.
get_ipython().run_line_magic('reload_ext', 'tensorboard')

# variáveis gerais para controle das simulações
MODEL_ARCH = 'LSTM'
LEARNING_RATE = 1.0e-4
PATIENCE = 30
TF_LOG = False # ativa o callback do tensorboard para geração de logs


#%% Classe Simulator
class Simulator:
    

    
    #%%Treino da RNA
    @tf.autograph.experimental.do_not_convert
    def train_model(self, cfg, autoregressive):
        """
        Função para treinamento das redes neurais

        Parameters
        ----------
        cfg : list
            Lista de parâmetros da rede neural e da simulação.

        Returns
        -------
        model: keras.model,
            Modelo treinado.
        test_X: np.array,
            Array de variáveis endógenas formatado para a avaliação da rede neural.
        test_y: np.array,
            Array de variáveis exógenas formatado para a avaliação da rede neural.
        scaler: sklearn.preprocessing.MinMaxScaler,
            Objeto de normalização dos dados da rede neural

        """    
        n_endog,n_steps, n_train_steps, n_features, n_nodes, n_epochs, n_batch = cfg
        
        global MODEL_ARCH
        global TENSOR_BOARD_LOG
        
        
        
        values = self.data.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # specify the number of lag quarters
        
            
        if autoregressive == False:
            #### é necessário reescalar os valores de acordo com a nova forma dos dados
                     
            # O MODELO NÃO É AUTORREGRESSIVO 
            train_X, train_y = train[1:6, :n_obs], train[0:1, -n_endog]
            test_X, test_y = test[1:6, :n_obs], test[0:1, -n_endog]
             
            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
            test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
                
                       
        else:
            # O MODELO É AUTORREGRESSIVO 
            # frame as supervised learning
            reframed = self.series_to_supervised(scaled, n_steps, 1)
      
            # split into train and test sets
            values = reframed.values
        
            n_obs = n_steps * n_features
        
            train = values[:n_train_steps, :]
            test = values[n_train_steps:, :]
            
            train_X, train_y = train[:, :n_obs], train[:, -n_features]
            test_X, test_y = test[:, :n_obs], test[:, -n_features]
             
            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
            test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
            
        # constrói e configura os parâmetros da rede neural
        model = Models()
        model.set_neurons(n_nodes)
        model.set_x_shape(test_X.shape[1])
        model.set_y_shape(test_X.shape[2])        
       
        if MODEL_ARCH == 'LSTM':
            model = model.lstm()
        elif MODEL_ARCH == 'LSTM-S':
            model = model.lstm_stacked()
        elif MODEL_ARCH == 'LSTM-B':
            model = model.lstm_bidirectional()
        elif MODEL_ARCH == 'GRU':
             model = model.gru()
        elif MODEL_ARCH == 'CNN-LSTM':
            model = model.cnn_lstm()
        else:
            print('**'*10)
            print("Erro! Modelo não identificado.")
            return
       
        batch_size=n_batch
         
        learning_rate=LEARNING_RATE
        
        callbacks = []
        
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
      
        model.compile(loss="mse", optimizer=opt, metrics=["mae"])
        
        
             
        patience_callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', 
                                                             patience=PATIENCE)
        
        
        callbacks.append(patience_callback)
        
        if TF_LOG == True:
            tensorboard = TensorBoard(log_dir="{}/{}-{}-{}-{}-{}-".format(LOGS_FLD,
                                                                          MODEL_ARCH,
                                                                          n_features, 
                                                      n_nodes, n_epochs, n_batch) + 
                                          datetime.now().strftime("%Y%m%d-%H%M%S"))  
            callbacks.append(tensorboard)    
        
        #print("\n --- Treinando o Modelo : ---\n")
        # fit network
        model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, 
                  validation_data=(test_X, test_y), verbose=0, shuffle=False, 
                  callbacks=callbacks)
         
        
        return model, test_X, test_y,scaler
    
    
    #%% Avaliação do Modelo
    @tf.autograph.experimental.do_not_convert
    def eval_model(self, cfg, autoregressive = True):
        """
        Função para avaliação das redes neurais com previsão dentro da amostra.

        Parameters
        ----------
        cfg : list
            Lista de parâmetros da rede neural e da simulação.

        Returns
        -------
        result_mean : np.array,
            Média dos resultados da previsão do modelo dentro da amostra.
        perf_mean: np.array,
            Média dos resultados da análise de desempenho do modelo.  
        cfg : list
            Lista de parâmetros da rede neural e da simulação.

        """  
        n_endog,n_steps, n_train_steps, n_features, n_nodes, n_epochs, n_batch = cfg
        
        global MODEL_ARCH
                
        resultado = []
        perf = np.zeros((self.n_rep))
          
        # Loop (TODO)
        # Vamos repetir o processo de treinamento por 20 vezes e armazenar todos os resultados, pois assim usaremos
        # diferentes amostras. Ao final, tiramos a média para encontrar as previsões. 
        # make a prediction
        print('##'*25)
        print(" --- Avaliando do Modelo : ---")
        print('##'*25)
        print('\n')
        print('##'*35)
        series_par = "{}-n_endog,{}-n_steps,{}-n_train_steps,{}-n_features".format(n_endog,
                                                                                         n_steps, 
                                                                                         n_train_steps, 
                                                                                         n_features)
        
        model_par =  "{}-n_nodes,{}-n_epochs,{}-n_batch".format(n_nodes, n_epochs, n_batch)                                                                                                          
                                                                                                                    
        
        print(f'## Parâmetros: \n{series_par}\n{model_par}\n ## ')
        print(datetime.now().strftime("%Y/%m/%d-%H:%M:%S\n"))
        
        #l = len(perf)
        
        #printProgressBar(0, l, prefix = 'Progress-evaluation:', suffix = 'Complete', length = 50)

        
        for i in tqdm(range(self.n_rep)):
          
           
          trainned_model, testx, testy, scaler = self.train_model(cfg, autoregressive)
          
          
          
          #modelo.model.model
          test_x = testx
          test_y = testy
          
          yhat = trainned_model.predict(test_x)
          #print(yhat.shape)
          test_x = test_x.reshape((test_x.shape[0], n_steps*n_features))
          
          # invert scaling for forecast
          inv_yhat = concatenate((yhat, test_x[:, -n_endog:]), axis=1)##
          yhat = scaler.inverse_transform(inv_yhat)
          yhat = yhat[:,0]
          
          
          resultado.append(yhat)
          
          #print(f'\nRepetição:{i+1}')
          #print(f'# épocas:({n_epochs}) # neurônios:({n_nodes}) # batch:({n_batch})')
          #print(f'loss:{round(modelo.history.history["loss"][-1],4)} - end val_loss: {round(modelo.history.history["val_loss"][-1],4)}\n')
          
          
          # invert scaling for actual
          test_y = test_y.reshape((len(test_y), 1))
          inv_y = concatenate((test_y, test_x[:, -n_endog:]), axis=1)
          y = scaler.inverse_transform(inv_y)
          y = y[:,0]
          
          perf[i] = self.performance(y,yhat)

          #printProgressBar(i + 1, l, prefix = 'Progress-evaluation:', suffix = 'Complete', length = 50)

          
        perf_mean = np.mean(perf)
        resultado = np.array(resultado) 
        
        # Loop para gerar as previsões finais
        result_mean = np.zeros((resultado.shape[1],1))
        for i in range(resultado.shape[1]):
          result_mean[i] = np.mean(resultado[:,i])
        
        if autoregressive == False:
            trainned_model.save("{}/model-{}-{}-{}-nar.h5".format(MODELS_FLD,MODEL_ARCH,series_par,model_par))  
        else:
            trainned_model.save("{}/model-{}-{}-{}.h5".format(MODELS_FLD,MODEL_ARCH,series_par,model_par))  
        
        
        return result_mean, perf_mean, cfg
      
    
    
    #%%Grid Search
    # Função para o Grid Search
    @tf.autograph.experimental.do_not_convert
    def grid_search(self):
        """
        Função que implementa o algoritmo de grid search para avaliação dos 
        modelos de redes neurais.

        Returns
        -------
        results : list,
            Lista com os resultados do algoritmo de grid search.
        errors : list,
            Lista com os erros dos modelos.
        hyperparams : list,
            Lista dos parâmetros dos modelos testados.

        """        
        errors = []
        results = []
        hyperparams = []
         
        # Gera os scores
        config = self.config
        l = len(config)
        # Initial call to print 0% progress
        print("\n")
        printProgressBar(0, l, prefix = 'Progress-grid search:', suffix = 'Complete', length = 50)
        
        #for i,cfg in enumerate(config):
        for i,cfg in enumerate(config):
          result,error,hyper = self.eval_model(cfg) 
          results.append(result)
          errors.append(error)
          hyperparams.append(hyper)
          # Update Progress Bar
          printProgressBar(i + 1, l, prefix = 'Progress-grid search:', suffix = 'Complete', length = 50)
          print("\n")
        
        # Ordena os hiperparâmetros pelo erro
        #errors.sort(key = lambda tup: tup[1])
        return results,errors, hyperparams
    
    
    #%%Seleciona o melhor resultado
    def best_result(self, error, result, hyperparms):
        """
        Função que escolhe o melhor resultado na análise de grid search.

        Parameters
        ----------
        errors : list,
            Lista com os erros dos modelos.
        results : list,
            Lista com os resultados do algoritmo de grid search.
        hyperparams : list,
            Lista dos parâmetros dos modelos testados.

        Returns
        -------
        best_fit : list,
            Lista com os resultados do modelo de melhor ajuste.
        best_hyper : list,
            Lista com os parâmetros do modelo de melhor ajuste.
        """      
        index = error.index(min(error))
        best_fit = result[index]
        best_hyper = hyperparms[index]
        print(f"Parâmetros do melhor modelo:{best_hyper}")
             
        print(f"Menor MSE: {round(min(error),3)}")
        return best_fit, best_hyper
    
    
    #%% Execução do Modelo com grid searh
    @tf.autograph.experimental.do_not_convert
    def run_grid_search(self):
        """
        Função que executa o algoritmo de grid search.

        Returns
        -------
        results : list,
            Lista com os resultados do algoritmo de grid search.
        errors : list,
            Lista com os erros dos modelos.
        hyperparams : list,
            Lista dos parâmetros dos modelos testados.

        """
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
        """
        Função que transforma os dados de séries temporais para em sequências 
        que possam ser entendidas pela rede neural. 

        Parameters
        ----------
        data : pd.DataFrame,
            Séries Temporais no formato de um DataFrame, onde a primeira coluna
            representa a variável eendógena e as seguintes as variáveis exógenas.
        n_in : int, optional
            Número de dados de entrada (t-n_in) para previsão do dado em t. The default is 1.
        n_out : int, optional
            Número de dados de saída (em t+n_out-1). The default is 1.
        dropnan : bool, optional
            Variável booleana. True para excluir NaN e False caso contrário. The default is True.

        Returns
        -------
        agg : pd.DataFrame,
            Séries na forma de sequências para interpretação da rede neural.

        """    
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
    def performance(self,y_true, y_pred):
        """
        Função que mede a capacidade preditiva para rede neural.

        Parameters
        ----------
        y_true : np.array,
            Array com os valores esperados.
        y_pred : np.array
            Array com os valores previstos.

        Returns
        -------
        mse : double,
            Erro quadrático médio.

        """    
        mse = mean_squared_error(y_true,y_pred)
        mape = mean_squared_error(y_true,y_pred)
        print('\nMSE:{}'.format(round(mse, 2))+
                       '-RMSE:{}'.format(round(np.sqrt(mse), 2))+
                      '-MAPE:{}\n'.format(round(mape, 2)))
        return mse

    #%% Realiza previsão fora da amostra 
    def forecast(self, data, n_inputs, n_predictions, model):
        """
        Função que realiza a previsão fora da amostra (projeção).

        Parameters
        ----------
        n_inputs : int,
            Número de dados utilizados na rede neural para o treinamento da 
            rede neural.
        n_predictions : int,
            Número de previsões fora da amostra.
        model : keras.models,
            Modelo de rede neural para realizar a previsão.

        Returns
        -------
        df_proj : pd.DataFrame,
            DataFrame com os resultados projetados (previsão fora da amostra)
        pred_list : list,
            Lista com os dados utilizados para a previsão.

        """            
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

    #%% Função que simula modelo
    @tf.autograph.experimental.do_not_convert
    def run_simulation(self):
        """
        Função que executa a simulação.

        Returns
        -------
        best_res : np.array,
            Array com o melhor resultado do modelo com o melhor desempenho.
        best_par : list,
            Lista dos parâmetros do modelo com o melhor desempenho.

        """    
        global MODEL_ARCH
        
        from numpy.random import seed
        seed(1)
           
        # Load the TensorBoard notebook extension.
        get_ipython().run_line_magic('reload_ext', 'tensorboard')
                 
        df = self.data
            
        var = ['Inv', 'Agr', 'Ind', 'Inf', 'Com']
           
        results, error, hyperparams = self.run_grid_search()
        
        # armazena o melhor resultado   
        best_res, best_par = self.best_result(error, results, hyperparams)
        
        # Escreve os resultados em arquivo
          
        with open('{}/best_{}_res.pkl'.format(PKL_FLD, MODEL_ARCH), 'wb') as fp:
            pickle.dump(best_res, fp) 
        
        
        with open('{}/best_{}_par.pkl'.format(PKL_FLD, MODEL_ARCH), 'wb') as fp:
            pickle.dump(best_par, fp)  
        
        
        # Visualiza a previsão do modelo - dados de teste com menor MSE
        plt.figure()
        
        # Série original
        plt.plot(df.index, 
                  df['Inv'].values,
                  label = 'Valores Observados',
                  color = 'Red')
        plt.plot(df.index[-len(best_res):], 
                 best_res,
                 label = 'Previsões com Modelo de Redes Neurais {}'.format(MODEL_ARCH), 
                 color = 'Black')
        plt.title('Previsões com Modelo de Redes Neurais Recorrentes')
        plt.xlabel('Ano')
        plt.ylabel('Investimento (%PIB)')
        plt.legend()
        plt.savefig('{}/predictions-{}'.format(FIGS_FLD,MODEL_ARCH))
        plt.show()    
        
        return best_res, best_par
    


    #%% Carrega dados do melhor modelo
    def load_best_model(self):
        """
        Função que carrega a arquitetura e os resultados da previsão do modelo 
        que apresentou melhor desempenho na avaliação. Lê dados dos arquivos
        gravados em disco.

        Returns
        -------
        best_model : keras.models, 
            Arquitetura do modelo de rede neural com melhor desempenho.
        best_res : np.array,
            Resultados da previsão do modelo de rede neural com melhor desempenho.
        n_inputs: int,
            Número de dados utilizados na rede neural para o treinamento da 
            rede neural.
            

        """ 
        global MODEL_ARCH

        with open ('{}/best_{}_par.pkl'.format(PKL_FLD,MODEL_ARCH), 'rb') as fp:
            best_par = pickle.load(fp)
       
        print(best_par)
        
                
        best_file = '{}/model-{}-{}-n_endog,{}-n_steps,{}-n_train_steps,{}-n_features-{}-n_nodes,{}-n_epochs,{}-n_batch.h5'.format(MODELS_FLD, MODEL_ARCH,
                                                                                                                                 best_par[0],
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
        
        with open ('{}/best_{}_res.pkl'.format(PKL_FLD, MODEL_ARCH), 'rb') as fp:
            best_res = pickle.load(fp)
    
        n_inputs = best_par[1]        
         
        return best_model, best_res, n_inputs

   
    #%% Class constructor
    def __init__(self, data, config):
        """
        Construtor do objeto Simulators.

        Parameters
        ----------
        data : pd.DataFrame,
            Séries Temporais no formato de um DataFrame, onde a primeira coluna
            representa a variável eendógena e as seguintes as variáveis exógenas.
        config : list
            Lista de parâmetros para a simulação.

        Returns
        -------
        None.

        """
        self.config = config
        
        
        config = np.array(config).reshape(len(config[0]),len(config))    
        
        self.n_endog = config[0]
        self.n_steps= config[1] 
        self.n_train_steps = config[2] 
        self.n_features = config[3] 
        self.n_nodes = config[4] 
        self.n_epochs = config[5] 
        self.n_batch = config[6]

        self.data = data
        self.n_rep = 10
        logging.info('## Redes Neurais construídas ##')    
        
    #%% Setters
    def set_n_endog(self, n_endog):
        """
        Função que configura o número de variárias endógenas.

        Parameters
        ----------
        n_endog : list ou int,
            Lista ou inteiro com o número de variáveis endógenas.

        Returns
        -------
        None.

        """
        self.n_endog = list(n_endog)
        
    def set_n_steps(self, n_steps):
        """
        Função que configura o número de intervalos da sequência para previsão.
        
        Parameters
        ----------
        n_step : list ou int,
            Lista ou inteiro com número de intervalos da sequência para previsão.

        Returns
        -------
        None.

        """
        self.n_steps = list(n_steps)
        
    def set_n_train_steps(self, n_train_steps):
        """
        Função que configura o número de intervalos para treino da rede neural.

        Parameters
        ----------
        n_train_steps : list ou int,
            Lista ou inteiro com o número de intervalos para treino da rede neural.

        Returns
        -------
        None.

        """
        self.n_train_steps = list(n_train_steps)
        
    def set_n_features(self, n_features):
        """
        

        Parameters
        ----------
        n_features : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.n_features= list(n_features)
        
    def set_n_nodes(self, n_nodes):
        """
        

        Parameters
        ----------
        n_nodes : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.n_nodes = list(n_nodes)
        
    def set_n_epochs(self, n_epochs):
        """
        

        Parameters
        ----------
        n_epochs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.n_epochs = list(n_epochs)
        
    def set_n_batch(self, n_batch):
        """
        

        Parameters
        ----------
        n_batch : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.n_batch = list(n_batch)
        
    def set_data(self, data):
        """
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.data = data
    
    def set_nrep(self, n_rep):
        """
        

        Parameters
        ----------
        n_rep : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.n_rep = n_rep
    
    def set_model_arch(self, model_arch):
        """
        

        Parameters
        ----------
        model_arch : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        global MODEL_ARCH 
        MODEL_ARCH = model_arch
        
    def get_model_arch(self):
        """
        

        Returns
        -------
        MODEL_ARCH : TYPE
            DESCRIPTION.

        """
        
        return MODEL_ARCH
    
# =============================================================================
# def main():
#     print('Simulação do Modelo')
#     
#     df = load_data()
#     var = ['Inv', 'Agr', 'Ind', 'Inf', 'Com']
#         
#     df = df[var]
#     
#     config = config_model(n_steps = [24,36],n_train_steps = [24,36],n_batch = [16,32])
# 
#     simulator = Simulator(df, config)
#     simulator.set_model_arch('LSTM')
#     #best_res_lstm, best_par_lstm = simulator.run_simulation()
#     
#     # Carrega o modelo que apresentou o melhor resultado na simulação
#     best_lstm_model, best_lstm_res, n_inputs = simulator.load_best_model()
#    
#     df_proj, pred_list = simulator.forecast(df,n_inputs, 24, best_lstm_model)
#     
# 
#     
#     # Cria Data Frame a partir da melhor previsão dentro da amostra
#     df_lstm = pd.DataFrame(best_lstm_res, columns=['Inv_lstm'])
#     df_lstm.index = df.index[-len(df_lstm):]  
#   
#     
#     # Cria Data Frame com todos os resultados    
#     df_proj = pd.concat([df_proj,df_lstm], axis=1)
# 
#     
#     # Gráfico com a série original, previsão do modelo dentro da amostra
#     # e previsão do modelo fora da amostra
#     plt.figure()
#     plt.plot(df_proj.index, df_proj['Inv'],
#              label = 'Valores Observados')
#     plt.plot(df_proj.index, df_proj['Inv_lstm'], 
#              label = 'Rede Neural - dentro da amostra',color='black')
#     plt.plot(df_proj.index, df_proj['Prediction'], 
#              label = 'Rede Neural - fora da amostra',color='blue')
# 
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=16)
#     plt.title('Previsões com Modelo de Redes Neurais Recorrentes')
#     plt.xlabel('Ano')
#     plt.ylabel('Investimento (%PIB)')
#     plt.legend()
#     plt.show()
# =============================================================================
    
    
    
#if __name__ == '__main__':
    #main()