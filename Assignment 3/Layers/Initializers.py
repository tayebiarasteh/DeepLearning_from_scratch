'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''


import numpy as np
from Layers.Base import *


class Constant(base_layer):
    '''
    self.shape = (output_size, input_size)
    '''
    def __init__(self, weight_initialization=0.1):
        '''
        :param weight_initialization: the constant value used for weight initialization.
        '''
        super().__init__()

        self.weight_initialization = weight_initialization

    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''
        return (np.zeros((fan_in, fan_out)) + self.weight_initialization)



class UniformRandom(base_layer):
    def __init__(self):
        super().__init__()

        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''
        return np.random.rand(fan_in, fan_out)



class Xavier(base_layer):
    def __init__(self):
        super().__init__()

        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''
        return np.random.normal(0, (2 / (fan_out + fan_in))**(1/2), weights_shape)



class He(base_layer):
    def __init__(self):
        super().__init__()

        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''
        return np.random.normal(0, (2 / fan_in)**(1/2), weights_shape)

