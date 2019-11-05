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
        self.data_layer = (0,0)
        self.loss_layer = np.empty()


    def forward(self):
        self.input_tensor = self.data_layer[0]
        self.label_tensor = self.data_layer[1]
        fcl = FullyConnected.FullyConnected(self.input_tensor.shape[0], self.input_tensor.shape[1])
        self.input_tensor = fcl.forward(self.input_tensor)
        relu = ReLU.ReLU()
        self.input_tensor = relu.forward(self.input_tensor)
        softmax = SoftMax.SoftMax()
        self.input_tensor = softmax.forward(self.input_tensor)
        loss = Loss.CrossEntropyLoss()
        self.input_tensor = loss.forward(self.input_tensor, self.label_tensor)

        return self.input_tensor


    def backward(self):
        loss = Loss.CrossEntropyLoss()
        self.error_tensor = loss.backward(self.label_tensor)
        softmax = SoftMax.SoftMax()
        self.error_tensor = softmax.backward(self.error_tensor)
        relu = ReLU.ReLU()
        self.error_tensor = relu.backward(self.error_tensor)
        fcl = FullyConnected.FullyConnected(self.input_tensor.shape[0], self.input_tensor.shape[1])
        self.error_tensor = fcl.backward(self.error_tensor)

        return self.error_tensor


    def append_trainable_layer(self, layer):
        pass


    def train(self,iterations):
        pass


