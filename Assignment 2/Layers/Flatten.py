import numpy as np


class Flatten:
    def __init__(self):
        pass


    def forward(self, input_tensor):
        '''reshapes and returns the input tensor.'''

        #saves the input tensor dimensions
        self.input_tensor_dim = (input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3])
        result = np.zeros((input_tensor.shape[0], input_tensor.shape[1]*input_tensor.shape[2]*input_tensor.shape[3]))
        for i in range(input_tensor.shape[0]):
            #first dimension is the batches.
            result[i] = input_tensor[i].flatten()
        return result



    def backward(self, error_tensor):
        '''reshapes back the input tensor.'''
        return error_tensor.reshape(self.input_tensor_dim)
