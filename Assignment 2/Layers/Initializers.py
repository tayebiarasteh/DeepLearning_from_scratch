import numpy as np


class Constant:
    def __init__(self, weight_initialization=0.1):
        '''
        :param weight_initialization: determines the constant value used for weight initialization.
        '''
        self.weight_initialization = weight_initialization


    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''

        pass




class UniformRandom:
    def __init__(self):
        pass


    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''

        pass



class Xavier:
    def __init__(self):
        pass


    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''

        pass



class He:
    def __init__(self):
        pass


    def initialize(self, weights_shape, fan_in, fan_out):
        ''''
        returns an initialized tensor of the desired shape.
        '''

        pass

