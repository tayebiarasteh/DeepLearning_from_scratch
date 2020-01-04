'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''

from Layers.Base import *
import numpy as np
import pdb
from Layers import Sigmoid, FullyConnected, TanH


class LSTM(base_layer):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        :input_size: denotes the dimension of the input vector
        :hidden_size: denotes the dimension of the hidden state.
        '''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros((self.hidden_size))
        self.cell_state = np.zeros((self.hidden_size))

        # Sets the boolean state representing whether the RNN
        # regards subsequent sequences as a belonging to the same long sequence.
        self._memorize = False

        self._optimizer = None #weight optimizer
        self._bias_optimizer = None
        self._gradient_weights = None
        self._weights = None

        # The weights are defined as the weights which are involved in calculating the
        # hidden state as a stacked tensor. E.g. if the hidden state is computed with
        # a single Fully Connected layer, which receives a stack of the hidden state
        # and the input tensor, the weights of this particular Fully Connected Layer,
        # are the weights considered to be weights for the whole class.
        # self._weights = None

        self.sigmoid1 = Sigmoid.Sigmoid()
        self.sigmoid2 = Sigmoid.Sigmoid()
        self.sigmoid3 = Sigmoid.Sigmoid()
        self.sigmoid4 = Sigmoid.Sigmoid()
        self.tanh1 = TanH.TanH()
        self.tanh2 = TanH.TanH()
        self.fully_middle = FullyConnected.FullyConnected(input_size=input_size + hidden_size ,
                                                          output_size=hidden_size)
        self.fully_out = FullyConnected.FullyConnected(input_size=hidden_size, output_size=output_size)


    def forward(self, input_tensor):
        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))

        if self._memorize == False:
            self.hidden_state = np.zeros((self.hidden_size))
            self.cell_state = np.zeros((self.hidden_size))

        # giving inputs sequentially
        for idx, batch in enumerate(input_tensor):
        # Concatenation of input and previous hidden state
            X_tilda = np.concatenate((self.hidden_state, batch))

            # Calculate forget gate
            f = self.fully_middle.forward(X_tilda)
            f = self.sigmoid1.forward(f)

            # Calculate input gate
            i = self.fully_middle.forward(X_tilda)
            i = self.sigmoid2.forward(i)

            # Calculate candidate
            C_tilda = self.fully_middle.forward(X_tilda)
            C_tilda = self.tanh1.forward(C_tilda)

            # Calculate memory state
            self.cell_state = f * self.cell_state + i * C_tilda

            # Calculate output gate
            o = self.fully_middle.forward(X_tilda)
            o = self.sigmoid3.forward(o)

            # Calculate hidden state
            self.hidden_state = o * self.tanh2.forward(self.cell_state)

            # Calculate logits
            y = self.fully_out.forward(self.hidden_state)
            y = self.sigmoid4.forward(y)

            output_tensor[idx] = y

        return output_tensor



    def backward(self, error_tensor):
        output_tensor = np.zeros((error_tensor.shape[0], self.input_size))

        # if self._memorize == False:
        #     self.hidden_state = np.zeros((self.hidden_size))
        #     self.cell_state = np.zeros((self.hidden_size))

        # giving inputs sequentially
        for idx, batch in enumerate(reversed(error_tensor)):
            # Calculate hidden state
            y = self.sigmoid4.backward(batch)
            self.hidden_state = self.fully_out.backward(y)

            # Calculate output gate
            o = self.hidden_state / self.tanh2.forward(self.cell_state)

            # Calculate hidden state
            self.cell_state = self.tanh2.backward(self.hidden_state / o)

            # backward output gate
            o = self.sigmoid3.backward(o)
            X_tilda = self.fully_middle.backward(o)

            # forward forget gate
            f = self.fully_middle.forward(X_tilda)
            f = self.sigmoid1.forward(f)

            # forward input gate
            i = self.fully_middle.forward(X_tilda)
            i = self.sigmoid2.forward(i)

            # forward candidate
            C_tilda = self.fully_middle.forward(X_tilda)
            C_tilda = self.tanh1.forward(C_tilda)

            # previous memory state
            self.cell_state = (self.cell_state - i * C_tilda) / f

            # de-concatenation of input and previous hidden state
            y = X_tilda[len(self.hidden_state):]

            # previous hidden state
            self.hidden_state = X_tilda[:len(self.hidden_state)]

            output_tensor[idx] = y

        return output_tensor


    def initialize(self, weights_initializer, bias_initializer):
        # output size of each fullyconnected is 4x hidden size
        self.weights = weights_initializer.initialize((self.input_size + self.hidden_size, 4 * self.hidden_size),
                                                      self.input_size + self.hidden_size, 4 * self.hidden_size)
        self.bias = bias_initializer.initialize((1, 4 * self.hidden_size), 1, 4 * self.hidden_size)
        self.weights = np.vstack((self.weights, self.bias))



    '''Properties'''

    @property
    def memorize(self):
        return self._memorize
    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
    @gradient_weights.deleter
    def gradient_weights(self):
        del self._gradient_weights

    @property
    def weights(self):
        return self._weights
    @weights.setter
    def weights(self, value):
        self._weights = value
    @weights.deleter
    def weights(self):
        del self._weights

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

    @property
    def bias_optimizer(self):
        return self._bias_optimizer
    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value
    @bias_optimizer.deleter
    def bias_optimizer(self):
        del self._bias_optimizer