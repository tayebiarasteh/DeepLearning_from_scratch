'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''


import numpy as np
from Layers.Base import *


class ReLU(base_layer):
    def __init__(self):
        super().__init__()

        pass

    def forward(self, input_tensor):
        '''
        returns the input_tensor for the next layer.
        '''
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor) #element-wise


    def backward(self, error_tensor):
        '''
        returns the error_tensor for the next layer.
        '''
        return np.where(self.input_tensor > 0, error_tensor, 0) #element-wise
