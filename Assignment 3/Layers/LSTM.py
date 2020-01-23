'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''

from Layers.Base import *
import numpy as np
import pdb
from Layers import Sigmoid, FullyConnected, TanH
import copy


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

        # activations
        self.o = []
        self.i = []
        self.f = []
        self.C_tilda = []
        self.cell_state_b = []
        self.hidden_state_b = []
        self.tanh2_out = []
        self.fully_middle_input = []

        if self._memorize == False:
            self.hidden_state = np.zeros((self.hidden_size))
            self.cell_state = np.zeros((self.hidden_size))

        self.cell_state_b.append(self.cell_state)
        self.hidden_state_b.append(np.zeros((self.hidden_size+1)))

        # giving inputs sequentially
        for idx, batch in enumerate(input_tensor):
        # Concatenation of input and previous hidden state
            X_tilda = np.concatenate([self.hidden_state, batch])
            # first fully connected layer
            fully_middle_out = self.fully_middle.forward(X_tilda)
            self.fully_middle_input.append((self.fully_middle.input_tensor))

            '''deconcatenating to 4 vectors'''
            # Calculate forget gate
            f = self.sigmoid1.forward(fully_middle_out[:fully_middle_out.shape[0]//4])
            self.f.append(f)

            # Calculate input gate
            i = self.sigmoid2.forward(fully_middle_out[fully_middle_out.shape[0]//4:fully_middle_out.shape[0]//2])
            self.i.append(i)

            # Calculate candidate
            C_tilda = self.tanh1.forward(fully_middle_out[fully_middle_out.shape[0]//2: 3*fully_middle_out.shape[0]//4])
            self.C_tilda.append(C_tilda)

            # Calculate memory state
            self.cell_state = f * self.cell_state + i * C_tilda
            self.cell_state_b.append(self.cell_state)

            # Calculate output gate
            o = self.sigmoid3.forward(fully_middle_out[3*fully_middle_out.shape[0]//4:])
            self.o.append(o)

            # tanh2 output
            tanh2_out = self.tanh2.forward(self.cell_state)
            self.tanh2_out.append(tanh2_out)

            # Calculate hidden state
            self.hidden_state = o * tanh2_out

            # Calculate logits
            y = self.fully_out.forward(self.hidden_state)
            self.hidden_state_b.append(self.fully_out.input_tensor)
            y = self.sigmoid4.forward(y)
            output_tensor[idx] = y

        self.output_tensor = output_tensor
        return output_tensor



    def backward(self, error_tensor):
        gradient_input = np.zeros((error_tensor.shape[0], self.input_size))

        # initializing the hidden and cell state gradients
        gradient_hidden = np.zeros((error_tensor.shape[0]+1, self.hidden_size))
        gradient_cell = np.zeros((error_tensor.shape[0]+1, self.hidden_size))
        gradient_weights_out = 0
        gradient_weights_middle = 0

        # giving inputs sequentially
        for idx in reversed(range(len(error_tensor))):

            # gradient of output w.r.t input
            self.sigmoid4.activation = self.output_tensor[idx]
            gradient_out_wrt_in = self.sigmoid4.backward(np.copy(error_tensor)[idx])
            self.fully_out.input_tensor = self.hidden_state_b[idx]
            gradient_out_wrt_in = self.fully_out.backward(gradient_out_wrt_in)
            gradient_weights_out += self.fully_out.gradient_weights
            # fully_out_weights += self.fully_out.weights

            # gradient summing
            out_hidden = gradient_hidden[idx] + gradient_out_wrt_in

            # gradient output gate
            o_gradient = np.copy(out_hidden) * self.tanh2_out[idx]
            self.sigmoid3.activation = self.o[idx]
            o_gradient = self.sigmoid3.backward(o_gradient)

            # gradient tanh2
            gradient_out_wrt_in_cell = np.copy(out_hidden) * self.o[idx]
            self.tanh2.activation = self.tanh2_out[idx]
            gradient_out_wrt_in_cell = self.tanh2.backward(gradient_out_wrt_in_cell)

            # gradient summing
            out_cell = gradient_out_wrt_in_cell + gradient_cell[idx+1]

            '''gradient of the summation'''
            # gradient candidate
            C_tilda_gradient = np.copy(out_cell) * self.i[idx]
            self.tanh1.activation = self.C_tilda[idx]
            C_tilda_gradient = self.tanh1.backward(C_tilda_gradient)

            # gradient input gate
            i_gradient = np.copy(out_cell) * self.C_tilda[idx]
            self.sigmoid2.activation = self.i[idx]
            i_gradient = self.sigmoid2.backward(i_gradient)

            # gradient cell
            gradient_cell[idx] = np.copy(out_cell) * self.f[idx]

            # gradient forget gate
            f_gradient = np.copy(out_cell) * self.cell_state_b[idx]
            self.sigmoid1.activation = self.f[idx]
            f_gradient = self.sigmoid1.backward(f_gradient)

            # concatenation for the fully connected
            self.fully_middle.input_tensor = self.fully_middle_input[idx]
            y = self.fully_middle.backward(np.concatenate([f_gradient, i_gradient, C_tilda_gradient, o_gradient]))
            gradient_weights_middle += self.fully_middle.gradient_weights

            gradient_hidden[idx-1] = y[:self.hidden_size]
            gradient_input[idx] = y[self.hidden_size:]

        if self._optimizer:
            self.fully_out.weights = self._optimizer2.calculate_update(self.fully_out.weights, gradient_weights_out)
            self.fully_middle.weights = self._optimizer.calculate_update(self.fully_middle.weights, gradient_weights_middle)
        self.final_gradient_weights = gradient_weights_middle
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
        return self.final_gradient_weights
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
        self._optimizer2 = copy.deepcopy(self._optimizer)
    @optimizer.deleter
    def optimizer(self):
        del self._optimizer