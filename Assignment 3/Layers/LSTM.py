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

        self._optimizer = None
        self._gradient_weights = 0

        # The weights are defined as the weights which are involved in calculating the
        # hidden state as a stacked tensor. E.g. if the hidden state is computed with
        # a single Fully Connected layer, which receives a stack of the hidden state
        # and the input tensor, the weights of this particular Fully Connected Layer,
        # are the weights considered to be weights for the whole class.
        self._weights = None

        self.sigmoid1 = Sigmoid.Sigmoid()
        self.sigmoid2 = Sigmoid.Sigmoid()
        self.sigmoid3 = Sigmoid.Sigmoid()
        self.sigmoid4 = Sigmoid.Sigmoid()
        self.tanh1 = TanH.TanH()
        self.tanh2 = TanH.TanH()
        self.fully_middle = FullyConnected.FullyConnected(input_size=input_size + hidden_size ,
                                                          output_size=4 * hidden_size)
        self.fully_out = FullyConnected.FullyConnected(input_size=hidden_size, output_size=output_size)


    def forward(self, input_tensor):
        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))

        if self._memorize == False:
            self.hidden_state = np.zeros((self.hidden_size))
            self.cell_state = np.zeros((self.hidden_size))

        # giving inputs sequentially
        for idx, batch in enumerate(input_tensor):
        # Concatenation of input and previous hidden state
            X_tilda = np.concatenate([self.hidden_state, batch])

            # first fully connected layer
            fully_out = self.fully_middle.forward(X_tilda)

            '''deconcatenating to 4 vectors'''
            # Calculate forget gate
            self.f = self.sigmoid1.forward(fully_out[:fully_out.shape[0]//4])

            # Calculate input gate
            self.i = self.sigmoid2.forward(fully_out[fully_out.shape[0]//4:fully_out.shape[0]//2])

            # Calculate candidate
            self.C_tilda = self.tanh1.forward(fully_out[fully_out.shape[0]//2: 3*fully_out.shape[0]//4])

            # Calculate memory state
            self.cell_state = self.f * self.cell_state + self.i * self.C_tilda

            # Calculate output gate
            self.o = self.sigmoid3.forward(fully_out[3*fully_out.shape[0]//4:])

            # tanh2 output
            self.tanh2_out = self.tanh2.forward(self.cell_state)

            # Calculate hidden state
            self.hidden_state = self.o * self.tanh2_out

            # Calculate logits
            y = self.fully_out.forward(self.hidden_state)
            y = self.sigmoid4.forward(y)

            output_tensor[idx] = y

        return output_tensor



    def backward(self, error_tensor):
        gradient_input = np.zeros((error_tensor.shape[0], self.input_size))


        # initializing the hidden and cell state gradients
        gradient_hidden = np.zeros((self.hidden_size))
        gradient_cell = np.zeros((self.hidden_size))

        # weights_temp = 0
        # gradient_weights_temp = np.zeros_like(self.fully_middle.gradient_weights)

        # giving inputs sequentially
        for idx, batch in enumerate(reversed(error_tensor)):

            # assign the optimizer of the LSTM to the optimizer of the fully connected
            if self._optimizer:
                self.fully_out.optimizer = self._optimizer
                self.fully_middle.optimizer = self._optimizer

            # gradient of output w.r.t input
            y = self.sigmoid4.backward(batch)
            gradient_out_wrt_in = self.fully_out.backward(y)
            # assign the weights of the fully connected to the weights of the LSTM
            # gradient_weights_temp[:self.hidden_size] += self.fully_out.gradient_weights[:-1]

            # gradient summing
            out_hidden = gradient_hidden + gradient_out_wrt_in

            # gradient output gate
            o_gradient = out_hidden * self.tanh2_out
            o_gradient = self.sigmoid3.backward(o_gradient)

            # gradient tanh2
            gradient_out_wrt_in_cell = out_hidden * self.o
            gradient_out_wrt_in_cell = self.tanh2.backward(gradient_out_wrt_in_cell)

            # gradient summing
            out_cell = gradient_out_wrt_in_cell + gradient_cell

            '''gradient of the summation'''
            # gradient candidate
            C_tilda_gradient = out_cell * self.i
            C_tilda_gradient = self.tanh1.backward(C_tilda_gradient)

            # gradient input gate
            i_gradient = out_cell * self.C_tilda
            i_gradient = self.sigmoid2.backward(i_gradient)

            # gradient cell
            gradient_cell = out_cell * self.f

            # gradient forget gate
            f_gradient = out_cell * self.cell_state
            f_gradient = self.sigmoid1.backward(f_gradient)

            # concatenation for the fully connected
            y = self.fully_middle.backward(np.concatenate([f_gradient, i_gradient, C_tilda_gradient, o_gradient]))

            # gradient_weights_temp = self.fully_middle.gradient_weights

            gradient_hidden = y[:self.hidden_size]
            y = y[self.hidden_size:]

            gradient_input[idx] = y

        # assign the weights of the fully connected to the weights of the LSTM
        # self._gradient_weights = gradient_weights_temp

        return gradient_input


    def initialize(self, weights_initializer, bias_initializer):
        self.fully_middle.initialize(weights_initializer, bias_initializer)
        self.fully_out.initialize(weights_initializer, bias_initializer)


    def calculate_regularization_loss(self, layer):
        r_loss = 0
        if hasattr(layer, 'optimizer'):
            if layer.optimizer:
                if layer.optimizer.regularizer:
                    r_loss += layer.optimizer.regularizer.norm(layer.weights)
        return r_loss



    '''Properties'''

    @property
    def memorize(self):
        return self._memorize
    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def gradient_weights(self):
        return self.fully_middle.gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, value):
        self.fully_middle.gradient_weights = value
    @gradient_weights.deleter
    def gradient_weights(self):
        del self.fully_middle.gradient_weights

    @property
    def weights(self):
        return self.fully_middle.weights
    @weights.setter
    def weights(self, value):
        self.fully_middle.weights = value
    @weights.deleter
    def weights(self):
        del self.fully_middle.weights

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    @optimizer.deleter
    def optimizer(self):
        del self._optimizer