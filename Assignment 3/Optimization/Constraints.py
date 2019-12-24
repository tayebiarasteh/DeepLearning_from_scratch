import numpy as np


class L2_Regularizer:
    def __init__(self, alpha):
        '''
        alpha: regularization weight
        '''
        self.alpha = alpha

    def calculate_gradient(self, weights):
        '''
        returns the shrinkage term in the backward pass
        '''
        return self.alpha * weights

    def norm(self, weights):
        '''
        Augments the loss with the L2 norm
        '''
        return self.alpha * np.sqrt((weights**2).sum())



class L1_Regularizer:
    def __init__(self, alpha):
        '''
        alpha: regularization weight
        '''
        self.alpha = alpha

    def calculate_gradient(self, weights):
        '''
        returns the shrinkage term in the backward pass
        '''
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        '''
        Augments the loss with the L1 norm
        '''
        return self.alpha * np.abs(weights).sum()

