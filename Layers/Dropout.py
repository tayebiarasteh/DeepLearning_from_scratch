'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''


from Layers.Base import *


class Dropout(base_layer):
    def __init__(self, probability):
        '''
        probability: the fraction of units to keep.
        '''
        super().__init__()
        self.probability = probability
        self.binary_mask = 0

    def forward(self, input_tensor):
        # in test time, no dropout!
        if self.phase == Phase.test:
            return input_tensor

        self.binary_mask = np.random.rand(input_tensor.shape[0], input_tensor.shape[1]) < self.probability
        output_tensor = input_tensor * self.binary_mask

        # inverted dropout
        output_tensor /= self.probability
        return output_tensor



    def backward(self, error_tensor):
        '''
        Back propagate the gradients through the neurons that were not killed off during the forward pass,
        as changing the output of the killed neurons doesn’t change the output, and thus their gradient is 0.
        Note: since we have inverted dropout, we don't need to multiply activations with p.
        '''
        return error_tensor * self.binary_mask
