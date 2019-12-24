'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''


import numpy as np


class base_optimizer:
    def __init__(self):
        self.regulizer = None

    def add_regularizer(self, regularizer):
        self.regulizer = regularizer


class Sgd(base_optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        '''
        :param weight_tensor: previous weight tensor
        :param gradient_tensor: dL/dW
        :return: the updated weights according to the basic gradient descent update scheme
        '''
        if self.regulizer:
            shrinkage = self.regulizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * shrinkage

        return weight_tensor - self.learning_rate*gradient_tensor


class SgdWithMomentum(base_optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        '''
        :param weight_tensor: previous weight tensor
        :param gradient_tensor: dL/dW
        :return: the updated weights according to the basic gradient descent update scheme
        '''
        self.v_k = self.momentum_rate*self.v_k - self.learning_rate*gradient_tensor

        if self.regulizer:
            shrinkage = self.regulizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * shrinkage

        return weight_tensor + self.v_k



class Adam(base_optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_k = 0
        self.r_k = 0
        self.v_hat = 0
        self.r_hat = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        '''
        :param weight_tensor: previous weight tensor
        :param gradient_tensor: dL/dW
        :return: the updated weights according to the basic gradient descent update scheme
        '''
        self.v_k = self.mu * self.v_k + (1 - self.mu) * gradient_tensor
        self.r_k = self.rho * self.r_k + (1 - self.rho) * gradient_tensor*gradient_tensor
        self.v_hat = self.v_k / (1 - self.mu**self.k)
        self.r_hat = self.r_k / (1 - self.rho**self.k)
        self.k += 1

        if self.regulizer:
            shrinkage = self.regulizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * shrinkage

        return weight_tensor - self.learning_rate * \
               ((self.v_hat + np.finfo(float).eps)/((np.sqrt(self.r_hat) + np.finfo(float).eps)))

