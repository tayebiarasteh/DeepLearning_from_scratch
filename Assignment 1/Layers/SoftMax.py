import numpy as np


class SoftMax:

    def __init__(self):
        pass


    def forward(self, input_tensor):
        '''
        returns the estimated class probabilities for each row representing an element of the batch.'''
        exp = np.exp(input_tensor - np.amax(input_tensor))
        sum_ = np.sum(exp,axis=1)
        self.pred = np.divide(exp, np.expand_dims(sum_, axis=1))
        return self.pred


    def backward(self, label_tensor):
        '''
        returns the error tensor for the next layer.'''
