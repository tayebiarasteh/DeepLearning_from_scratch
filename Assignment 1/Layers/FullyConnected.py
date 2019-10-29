import numpy as np
import pdb

class FullyConnected:

    def __init__(self, input_size = np.random.uniform(0,1,1), output_size = np.random.uniform(0,1,1)):
        '''
        :param output_size: a parameter of the layer specifying the row dimensionality of the output.
        '''
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = 0


    def forward(self, input_tensor):
        '''
        :param input_tensor: a matrix with columns of arbitrary dimensionality input_size
            and rows of size batch_size representing the number of inputs processed simultaneously.
        :return: the input_tensor for the next layer.
        '''
        self.batch_size = input_tensor.shape[0]

        # random initialization of weights
        weight_tensor = np.random.rand(self.input_size+1, self.output_size ) # W'

        # adding an extra row of ones according to the slide 24 of "Neural Networks"
        input_tensor = np.concatenate((input_tensor, np.ones([self.batch_size, 1])), axis=1) # X'

        # Eq. 6 of "Neural Networks"
        return np.matmul(input_tensor, weight_tensor)


    def backward(self, error_tensor):
        '''
        :param error_tensor:
        :return: the error tensor for the next layer.
        '''
        return error_tensor



    def set_optimizer(self, optimizer):
        pass


    # property optimizer: sets and returns the protected member _optimizer for this layer.
    @property
    def optimizer(self):
        """I'm the 'optimizer' property."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer
