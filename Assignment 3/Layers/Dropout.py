'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''


import numpy as np
from Layers.Base import *


class Dropout(base_layer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):

        return input_tensor


    def backward(self, error_tensor):

        return error_tensor