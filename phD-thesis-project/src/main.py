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
from lib import *
from simulator import *



#%% Configuração do modelo
# Lista de hiperparâmetros que serão testados
def config_model(n_exog = [4], n_steps = [8], n_train_steps = [24], n_features = [5],
                 n_nodes = [150,300],n_epochs = [100],n_batch = [128] ):
    """
    

    Parameters
    ----------
    n_exog : list, optional
        Número de variáveis exógenas do modelo. The default is [4].
    n_steps : list, optional
        Número de intervalos de tempo anteriores para a previsão. The default is [8].
    n_train_steps : list, optional
        Número de intervalos de tempo para treino da rede neural. The default is [24].
    n_features : list, optional
        Número total de variáveis exógenas e endógenas. The default is [5].
    n_nodes : list, optional
        Número de neurônios da rede neural. The default is [150,300].
    n_epochs : list, optional
        Número máximo de épocas de treinamento. The default is [100].
    n_batch : list, optional
        Número total de batches da rede neural. The default is [128].

    Returns
    -------
    configs: list
    Lista de parâmetros para a simulação individual o para realização
    de grid search

    """
    
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

def run_lstm(data):
    """
    Função que executa a simulação com a arquitetura de rede LSTM.

    Returns
    -------
    None.

    """
         
    config = config_model(n_steps = [24,36],n_train_steps = [24,36],n_batch = [16,32])

    simulator = Simulator(data, config)
    simulator.set_model_arch('LSTM')
    #best_res, best_par = simulator.run_simulation()
    
    # Carrega o modelo que apresentou o melhor resultado na simulação
    best_model, best_res, n_inputs = simulator.load_best_model()
   
    df_proj, pred_list = simulator.forecast(data, n_inputs, 24, best_model)
    

    model_name = 'Inv_{}'.format(simulator.get_model_arch())    

    # Cria Data Frame a partir da melhor previsão dentro da amostra
    df_NN = pd.DataFrame(best_res, columns=[model_name])
    df_NN.index = data.index[-len(df_NN):]     
  
    
    # Cria Data Frame com todos os resultados    
    df_proj = pd.concat([df_proj,df_NN], axis=1)
    
    plot_results(df_proj, model_name)
    
    return

def run_lstm_bidirecccional(data):
    """
    Função que executa a simulação com a arquitetura de rede LSTM-Bidirecional.

    Returns
    -------
    None.

    """
    config = config_model(n_steps = [24,36],n_train_steps = [24,36],n_batch = [16,32])

    simulator = Simulator(data, config)
    simulator.set_model_arch('LSTM-B')
    best_res, best_par = simulator.run_simulation()
    
    # Carrega o modelo que apresentou o melhor resultado na simulação
    best_model, best_res, n_inputs = simulator.load_best_model()
   
    df_proj, pred_list = simulator.forecast(data, n_inputs, 24, best_model)
    

    model_name = 'Inv_{}'.format(simulator.get_model_arch())    

    # Cria Data Frame a partir da melhor previsão dentro da amostra
    df_NN = pd.DataFrame(best_res, columns=[model_name])
    df_NN.index = data.index[-len(df_NN):]      
  
    
    # Cria Data Frame com todos os resultados    
    df_proj = pd.concat([df_proj,df_NN], axis=1)
    
    plot_results(df_proj, model_name)    
    
    return

def run_lstm_stacked(data):
    """
    Função que executa a simulação com a arquitetura de rede LSTM empilhado.

    Returns
    -------
    None.

    """
    config = config_model(n_steps = [24,36],n_train_steps = [24,36],n_batch = [16,32])

    simulator = Simulator(data, config)
    simulator.set_model_arch('LSTM-S')
    #best_res, best_par = simulator.run_simulation()
    
    # Carrega o modelo que apresentou o melhor resultado na simulação
    best_model, best_res, n_inputs = simulator.load_best_model()
   
    df_proj, pred_list = simulator.forecast(data, n_inputs, 24, best_model)
    

    
    model_name = 'Inv_{}'.format(simulator.get_model_arch())    

    # Cria Data Frame a partir da melhor previsão dentro da amostra
    df_NN = pd.DataFrame(best_res, columns=[model_name])
    df_NN.index = data.index[-len(df_NN):]       
  
    
    # Cria Data Frame com todos os resultados    
    df_proj = pd.concat([df_proj,df_NN], axis=1)
    
    plot_results(df_proj, model_name)    
    
    return

def run_gru(data):
    """
    Função que executa a simulação com a arquitetura de rede GRU.

    Returns
    -------
    None.

    """
    
    config = config_model(n_steps = [24,36],n_train_steps = [24,36],n_batch = [16,32])

    simulator = Simulator(data, config)
    simulator.set_model_arch('GRU')
    best_res, best_par = simulator.run_simulation()
    
    # Carrega o modelo que apresentou o melhor resultado na simulação
    best_model, best_res, n_inputs = simulator.load_best_model()
   
    df_proj, pred_list = simulator.forecast(data, n_inputs, 24, best_model)
    

    
    model_name = 'Inv_{}'.format(simulator.get_model_arch())    

    # Cria Data Frame a partir da melhor previsão dentro da amostra
    df_NN = pd.DataFrame(best_res, columns=[model_name])
    df_NN.index = data.index[-len(df_NN):]       
  
    
    # Cria Data Frame com todos os resultados    
    df_proj = pd.concat([df_proj,df_NN], axis=1)
    
    plot_results(df_proj, model_name)    

    return
@tf.autograph.experimental.do_not_convert
def run_cnnn_lstm(data):
    """
    Função que executa a simulação com a arquitetura de rede CNN-LSTM.

    Returns
    -------
    None.

    """
    config = config_model(n_steps = [24,36],n_train_steps = [24,36],n_batch = [16,32])

    simulator = Simulator(data, config)
    simulator.set_model_arch('CNN-LSTM')
    best_res, best_par = simulator.run_simulation()
    
    # Carrega o modelo que apresentou o melhor resultado na simulação
    best_model, best_res, n_inputs = simulator.load_best_model()
   
    df_proj, pred_list = simulator.forecast(data, n_inputs, 24, best_model)
    
   
    model_name = 'Inv_{}'.format(simulator.get_model_arch())    

    # Cria Data Frame a partir da melhor previsão dentro da amostra
    df_NN = pd.DataFrame(best_res, columns=[model_name])
    df_NN.index = data.index[-len(df_NN):]   
  
 
    # Cria Data Frame com todos os resultados    
    df_proj = pd.concat([df_proj,df_NN], axis=1)
    

    
    plot_results(df_proj, model_name)
    
    
    
    return

#%% Visualização dos resultados da simulação
def plot_results(data, model_name):
    """
     Função para visualização da série original, previsão do modelo dentro da amostra
     e previsão do modelo fora da amostra

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    model_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    plt.figure()
    plt.plot(data.index, data['Inv'],
             label = 'Valores Observados')
    plt.plot(data.index, data[str(model_name)], 
             label = 'Rede Neural - dentro da amostra',color='black')
    plt.plot(data.index, data['Prediction'], 
             label = 'Rede Neural - fora da amostra',color='blue')

    plt.title('Previsões com Modelo de Redes Neurais Recorrentes')
    plt.xlabel('Ano')
    plt.ylabel('Investimento (%PIB)')
    plt.legend()
    plt.savefig('{}/forecast-{}'.format(FIGS_FLD,model_name))
    plt.show()

#%% Função principal
def main():
    """
    Função principal para execução do código.

    Returns
    -------
    None.

    """
    # abre arquivo para gravar a saída da simulação em arquivo
    #sys.stdout = open('{}/{}-app.log'.format(LOGS_FLD,datetime.now().strftime("%Y%m%d-%H%M%S")), 'w')
    logging.basicConfig(filename='{}/{}-app.log'.format(LOGS_FLD,datetime.now().strftime("%Y%m%d-%H%M%S")),
                        level=logging.INFO)
    logging.info('## Início do log ##')
    
    
    # Versões dos pacotes usados neste código
    get_ipython().run_line_magic('reload_ext', 'watermark')
    get_ipython().run_line_magic('watermark', '-a "Carlos Eduardo Veras Neves" --iversions')
    

    # Prepara diretórios para armazenar arquivos gerados pela simulação
    makedirs(MODELS_FLD) 
    makedirs(FIGS_FLD)
    makedirs(LOGS_FLD)
    makedirs(PKL_FLD)

    print("**"*25)
    print('--- Início da Simulação: ---')
    print("**"*25)
    
    df = load_data()
    var = ['Inv', 'Agr', 'Ind', 'Inf', 'Com']
        
    df = df[var]
    
    
    run_lstm(df)
    #run_lstm_stacked(df)
    #run_lstm_bidirecccional(df)
    #run_gru(df)
    #run_cnnn_lstm(df)
    
    
    
    
    print("**"*25)
    print('-- Fim da Simulação: --')
    print("**"*25)
    
    logging.info('## Fim do log ##')

    #sys.stdout.close()




    
if __name__ == '__main__':
	main()
