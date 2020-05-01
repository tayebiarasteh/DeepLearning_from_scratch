'''
Created on December 2019.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/starasteh/
'''

import numpy as np
from Optimization import *
from Layers.Base import *


class FullyConnected(base_layer):

    def __init__(self, input_size = np.random.uniform(0,1,1), output_size = np.random.uniform(0,1,1)):
        '''
        :param output_size: a parameter of the layer specifying the row dimensionality of the output.
        '''
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # random initialization of weights (including bias in the weight matrix).
        self.weights = np.random.rand(self.input_size + 1, self.output_size ) # W'
        self._optimizer = None
        self._gradient_weights = np.zeros_like(self.weights)


    def forward(self, input_tensor):
        '''
        :param input_tensor: a matrix with columns of arbitrary dimensionality input_size
            and rows of size batch_size representing the number of inputs processed simultaneously.
        :return: the input_tensor for the next layer.
        '''
        if input_tensor.ndim == 1:
            input_tensor = np.concatenate((input_tensor, np.ones((1))))  # X'
            self.input_tensor = input_tensor
            return np.matmul(input_tensor, self.weights)

        self.batch_size = input_tensor.shape[0]
        # adding an extra row of ones as bias
        input_tensor = np.concatenate((input_tensor, np.ones([self.batch_size, 1])), axis=1) # X'
        self.input_tensor = input_tensor
        return np.matmul(input_tensor, self.weights)


    def backward(self, error_tensor):
        '''
        :return: the error tensor for the next layer.
        '''
        gradient_input = np.matmul(error_tensor, self.weights.T)

        if self.input_tensor.ndim == 1:
            in_tensor = np.expand_dims(self.input_tensor, axis=0)
            self.gradient_weights = np.matmul(in_tensor.T, np.expand_dims(error_tensor, axis=0))
        else:
            self.gradient_weights = np.matmul(self.input_tensor.T, error_tensor)

        # updating the weights
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        if error_tensor.ndim == 1:
            return gradient_input[:-1]

        #removing the bias column from the input.
        return gradient_input[:,:-1]


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((self.weights, self.bias))


    def set_optimizer(self, optimizer):
        self._optimizer = optimizer


    '''property optimizer: sets and returns the protected member _optimizer for this layer.'''
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

    '''property gradient_weights: sets and returns the protected member _gradient_weights for this layer.'''
    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
    @gradient_weights.deleter
    def gradient_weights(self):
        del self._gradient_weights