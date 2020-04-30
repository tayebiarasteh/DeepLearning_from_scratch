'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/starasteh
'''

import numpy as np
from Layers import *
from Optimization import *
import copy

class NeuralNetwork:
    '''
    The Neural Network defines the whole architecture by containing all its layers from the input
    to the loss. This Network manages the testing and the training, that means it calls all forward
    methods passing the data from the beginning to the end, as well as the optimization by calling
    all backward passes afterwards.
    '''
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        '''
        loss: A list which will contain the loss value for each iteration after calling train.
        layers: A list which will hold the architecture.
        data_layer: a member, which will provide input data and labels upon calling forward() on it.
        loss_layer: Loss functions of the network. A member referring to the special layer providing loss and prediction
        '''
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = []
        self.loss_layer = []
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = Base.Phase.train


    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.forward()
        r_loss = 0

        for layer in self.layers:
            layer.phase = Base.Phase.train
            self.input_tensor = layer.forward(self.input_tensor)
            if hasattr(layer, 'optimizer'):
                if layer.optimizer:
                    if layer.optimizer.regularizer:
                        r_loss += layer.optimizer.regularizer.norm(layer.weights)

        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        self.loss.append(loss + r_loss)
        return loss + r_loss


    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)


    def append_trainable_layer(self, layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)


    def train(self, iterations):
        self.phase = Base.Phase.train

        for i in range(iterations):
            loss = self.forward()
            if (i+1)%200 == 0:
                print("training iteration",  str(i+1) + ":", 'loss =', loss)
            self.backward()


    def test(self, input_tensor):
        '''
        propagates the input tensor through the network
        and returns the prediction of the last layer.
        '''
        self.phase = Base.Phase.test

        for layer in self.layers:
            layer.phase = Base.Phase.test
            input_tensor = layer.forward(input_tensor)
        return input_tensor



    @property
    def phase(self):
        return self._phase
    @phase.setter
    def phase(self, value):
        self._phase = value
