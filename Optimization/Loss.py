'''
Created on December 2019.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
'''

import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        pass


    def forward(self, input_tensor, label_tensor):
        '''
        :param input_tensor: yk_hat
        :param label_tensor: yk
        :return: Computes the Loss value according the CrossEntropy Loss formula accumulated over the batch.
        '''
        # Eq. 12
        self.input_tensor = input_tensor
        input_tensor = input_tensor[label_tensor==1]
        temp1 = input_tensor + np.finfo(float).eps
        temp2 = np.log(temp1) * (-1)
        return np.sum(temp2)

    def backward(self, label_tensor):
        '''
        returns the error tensor for the next layer.
        Eq. 13: En = -y/y_hat
        '''
        return (-1) * label_tensor / self.input_tensor