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
        print(loss)
        # self.input_tensor = self.data_layer[0]
        # self.label_tensor = self.data_layer[1]
        # fcl = FullyConnected.FullyConnected(self.input_tensor.shape[0], self.input_tensor.shape[1])
        # self.input_tensor = fcl.forward(self.input_tensor)
        # relu = ReLU.ReLU()
        # self.input_tensor = relu.forward(self.input_tensor)
        # softmax = SoftMax.SoftMax()
        # self.input_tensor = softmax.forward(self.input_tensor)
        # loss = Loss.CrossEntropyLoss()
        # self.input_tensor = loss.forward(self.input_tensor, self.label_tensor)

        return loss


    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            print(layer)
            error_tensor = layer.backward(error_tensor)

        # loss = Loss.CrossEntropyLoss()
        # self.error_tensor = loss.backward(self.label_tensor)
        # softmax = SoftMax.SoftMax()
        # self.error_tensor = softmax.backward(self.error_tensor)
        # relu = ReLU.ReLU()
        # self.error_tensor = relu.backward(self.error_tensor)
        # fcl = FullyConnected.FullyConnected(self.input_tensor.shape[0], self.input_tensor.shape[1])
        # self.error_tensor = fcl.backward(self.error_tensor)

        #return self.error_tensor


    def append_trainable_layer(self, layer):
        self.layers.append(layer)


    def train(self,iterations):
        for i in range(iterations):
            self.forward()
            self.backward()

    def test(self,input_tensor):
        self.forward()
        return self.prediction


