import numpy as np


class SoftMax:

    def __init__(self):
        pass


    def forward(self, input_tensor):
        '''
        returns the estimated class probabilities for each row representing an element of the batch.'''



        X_hat = input_tensor - np.max(input_tensor) #to increase numerical stability
        # Eq. 10
        nom = np.exp(X_hat)
        den = np.sum(nom, axis=1)
        self.pred = np.divide(nom, np.expand_dims(den, axis=1)) # np.expand_dims() because it should be a column vector.
        return self.pred
       


    def backward(self, error_tensor):
        '''
        returns the error tensor for the next layer.'''

        # Eq. 11
        temp1 = np.sum(error_tensor * self.pred, axis=1)
        temp1 = np.expand_dims(temp1, axis=1) # np.expand_dims() because it should be a column vector.
        temp2 = error_tensor - temp1
        return self.pred * temp2

