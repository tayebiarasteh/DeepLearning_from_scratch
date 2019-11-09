import numpy as np
from Layers import *
from Optimization import *


class NeuralNetwork:

    def __init__(self, optimizer):
        '''
        :param optimizer:
        loss: A list which will contain the loss value for each iteration after calling train.
        layers: A list which will hold the architecture.
        data_layer: a member, which will provide input data and labels.
        loss_layer: a member referring to the special layer providing loss and prediction
        '''
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = []
        self.loss_layer = []


    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.forward()
        input = self.input_tensor
        for layer in self.layers:
            input = layer.forward(input)
        self.prediction = input
        loss = self.loss_layer.forward(input, self.label_tensor)
        self.loss.append(loss)

        return loss


    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)


    def append_trainable_layer(self, layer):
        # deep copy??? not working with np.copy(self.optimizer)
        layer.optimizer = self.optimizer
        self.layers.append(layer)


    def train(self,iterations):
        for i in range(iterations):
            loss = self.forward()
            if (i+1)%50 == 0:
                print("training iteration",  str(i+1) + ":", 'loss =', loss)
            self.backward()


    def test(self,input_tensor):
        '''
        propagates the input tensor through the network
        and returns the prediction of the last layer.
        '''
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor


