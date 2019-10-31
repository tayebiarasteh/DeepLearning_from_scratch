import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        pass


    def forward(self, input_tensor, label_tensor):
        '''
        Computes the Loss value according the CrossEntropy Loss formula accumulated over the batch.'''

        # Eq. 12
        temp1 = label_tensor + np.finfo(float).eps
        temp2 = np.log(temp1) * (-1)
        return np.sum(temp2, axis=0)

    def backward(self, label_tensor):
        '''
        returns the error tensor for the next layer.'''
