#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:39:14 2020

@author: Carlos Eduardo Veras Neves - PhD candidate
University of Brasilia - UnB
Department of Economics - Applied Economics

Thesis Supervisor: Geovana Lorena Bertussi, PhD.
Title:Professor of Department of Economics - UnB

Classe para construção das diversas arquiteturas de redes neurais.
"""
from lib import *

class Models:

    MODEL_ARCH = ''

    #%% Constrói o modelo LSTM
    def lstm(self):
        """
        Função que constrói o modelo baseado em neurônios LSTM.
        Usa a arquitetura CuDNN para execução mais rápida.

       Returns
       -------
       model : keras.model,
           Modelo com arquitetura baseada em neurônios LSTM.

        """


        global MODEL_ARCH

        # nome da arquitetura modelo
        MODEL_ARCH = 'LSTM'

        #número de neurônios nas camadas ocultas
        hidden_nodes = int(self.neurons*2/3)
        dropout = 0.2

        # modelo de rede de acordo com a configuração
        model = keras.Sequential()

        # CUDNN LSTM implementation
        model.add(keras.layers.LSTM(units = self.neurons, activation = 'tanh',
                             recurrent_activation = 'sigmoid',
                   recurrent_dropout = 0,unroll = False,
                   use_bias = True,
                   input_shape=(self.x_shape, self.y_shape)))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(hidden_nodes, activation = 'relu'))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1))

        return model

    #%% Constrói o modelo LSTM - stacked
    def lstm_stacked(self):
        """
        Função que constrói o modelo baseado em neurônios LSTM empilhados.

        Returns
        -------
        model : keras.model,
            Modelo com arquitetura baseada em neurônios LSTM empilhados.


        """
        global MODEL_ARCH

        # nome da arquitetura modelo
        MODEL_ARCH ='LSTM-S'


        #número de neurônios nas camadas ocultas
        hidden_nodes = int(n_nodes*2/3)
        dropout = 0.2
        # modelo de rede de acordo com a configuração
        model = keras.Sequential()


        # Stacked LSTM model
        model.add(keras.layers.LSTM(units = neurons, activation = 'relu',recurrent_activation = 'sigmoid',
                          recurrent_dropout = 0,unroll = False, use_bias = True, return_sequences = True,
                         input_shape=(x_shape, y_shape)))
        # adicionar camada LSTM para usar o disposito de recorrência
        model.add(keras.layers.LSTM(units = neurons, activation = 'relu'))
        model.add(keras.layers.Dense(hidden_nodes, activation = 'tanh'))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1))

        return model

    #%% Constrói o modelo LSTM - bidirecional
    def lstm_bidirectional(self):
        """
        Função que constrói o modelo baseado em neurônios LSTM-Bidirecional

        Returns
        -------
        model : keras.model,
            Modelo com arquitetura baseada em neurônios LSTM-Bidirecional.


        """
        global MODEL_ARCH

        # nome da arquitetura modelo
        MODEL_ARCH = 'LSTM-B'

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

        return model


    #%% Constrói o modelo LSTM - bidirecional
    def gru(self):
        """
        Função que constrói o modelo baseado em neurônios GRU.

        Returns
        -------
        model : keras.model,
            Modelo com arquitetura baseada em neurônios GRU.


        """
        global MODEL_ARCH
        # nome da arquitetura modelo
        MODEL_ARCH = 'GRU'

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

        return model

    def set_x_shape(self,x_shape):
        """


        Parameters
        ----------
        x_shape : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.x_shape = x_shape

    def set_y_shape(self,y_shape):
        """


        Parameters
        ----------
        y_shape : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.y_shape = y_shape

    def set_neurons(self,neurons):
        """


        Parameters
        ----------
        neurons : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.neurons = neurons

    def get_model_name(self):
        """


        Returns
        -------
        None.

        """
        return self.model_name


    #%% Class constructor
    def __init__(self):

            self.neurons = 0
            self.x_shape = 0
            self.y_shape = 0

