import numpy as np
import pdb


class Flatten:
    def __init__(self):
        pass


    def forward(self, input_tensor):
        '''reshapes and returns the input tensor.'''
        result = np.zeros((input_tensor.shape[0], input_tensor.shape[1]*input_tensor.shape[2]*input_tensor.shape[3]))
        for i in range(input_tensor.shape[0]):
            result[i] = input_tensor[i].flatten()
        return result



    def backward(self, error_tensor):
        pass
