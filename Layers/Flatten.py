'''
Created on December 2019.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/starasteh/
'''

import numpy as np
from Layers.Base import *


class Flatten(base_layer):
    def __init__(self):
        super().__init__()
        pass


    def forward(self, input_tensor):
        '''reshapes and returns the input tensor.'''

        if len(input_tensor.shape) ==2:
            self.input_tensor_dim = input_tensor.shape
            return input_tensor

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
