import numpy as np
from Optimization import *
import pdb

class FullyConnected:

    def __init__(self, input_size = np.random.uniform(0,1,1), output_size = np.random.uniform(0,1,1)):
        '''
        :param output_size: a parameter of the layer specifying the row dimensionality of the output.
        '''
        self.input_size = input_size
        self.output_size = output_size

        # random initialization of weights (including bias in the weight matrix).
        self.weights = np.random.rand(self.input_size + 1, self.output_size ) # W'
        # optimizer temporary constructor
        self._optimizer = Optimizers.Sgd(0)
        # temporary constructor
        self._gradient_weights = np.zeros_like(self.weights)


    def forward(self, input_tensor):
        '''
        :param input_tensor: a matrix with columns of arbitrary dimensionality input_size
            and rows of size batch_size representing the number of inputs processed simultaneously.
        :return: the input_tensor for the next layer.
        '''
        self.batch_size = input_tensor.shape[0]
        self.input_tensor = input_tensor
        # adding an extra row of ones according to the slide 24 of "Neural Networks"
        input_tensor = np.concatenate((input_tensor, np.ones([self.batch_size, 1])), axis=1) # X'
        # Eq. 6 of "Neural Networks"
        return np.matmul(input_tensor, self.weights)


    def backward(self, error_tensor):
        '''
        :param error_tensor:
        :return: the error tensor for the next layer.
        '''
        # updating the weights
        # pdb.set_trace()
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        # self.gradient_weights = np.matmul(self.input_tensor.T, error_tensor)
        #gradient w.r.t. X (remove bias)
        input_tensor = np.matmul(error_tensor, self.weights.T)

        #removing the bias column from the input.
        return input_tensor[:,:-1]



    '''property optimizer: sets and returns the protected member _optimizer for this layer.'''
    @property
    def optimizer(self):
        """I'm the 'optimizer' property."""
        return self._optimizer


    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

        # the test file is not correct here, because it's not considering the bias in the 2nd dim of the input.
        input_tensor = np.matmul(error_tensor, self.weights.T)
        return input_tensor[:,:-1]


    @optimizer.deleter
    def optimizer(self):
        del self._optimizer


    '''property gradient_weights: sets and returns the protected member _gradient_weights for this layer.'''
    @property
    def gradient_weights(self):
        """I'm the 'gradient_weights' property."""
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @gradient_weights.deleter
    def gradient_weights(self):
        del self._gradient_weights
