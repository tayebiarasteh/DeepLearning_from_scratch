import numpy as np


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
        self.data_layer = np.empty()