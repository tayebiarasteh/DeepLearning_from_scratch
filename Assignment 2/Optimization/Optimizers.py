import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        '''
        :param weight_tensor: previous weight tensor
        :param gradient_tensor: dL/dW
        :return: the updated weights according to the basic gradient descent update scheme
        '''
        return weight_tensor - self.learning_rate*gradient_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate


    def calculate_update(self, weight_tensor, gradient_tensor):
        # v_k_previous = self.learning_rate*gradient_tensor
        v_k = self.momentum_rate - self.learning_rate*gradient_tensor
        return weight_tensor + v_k



class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho


    def calculate_update(self, weight_tensor, gradient_tensor):
        pass
