import numpy as np
from scipy import signal
import pdb


class Conv:
    def __init__(self, stride_shape=np.random.uniform(0,1,1)[0],
                 convolution_shape=np.random.uniform(0,1,2), num_kernels=np.random.uniform(0,1,1)[0]):
        '''
        :param stride_shape: can be a single value or a tuple. The latter allows for different strides
            in the spatial dimensions.

        :param convolution_shape: determines whether this objects provides a 1D or a 2D convolution layer.
            For 1D, it has the shape [c, m], whereas for 2D, it has the shape [c, m, n],
            where c represents the number of input channels, and m, n represent the spacial extent of the filter kernel.

        :param num_kernels: an integer value
        '''
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        # 2D convolution layer (c, patch[0], patch[1])
        # c: channels
        if len(convolution_shape)==3:
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2])
        # 1D convolution layer (c, patch[0])
        elif len(convolution_shape)==2:
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1])
        self.bias = np.random.rand(num_kernels)
        self._optimizer = None #weight optimizer
        self._bias_optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None



    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        if len(self.convolution_shape)==3:
            result = np.zeros((input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2]/self.stride_shape[0])),
                               int(np.ceil(input_tensor.shape[3]/self.stride_shape[1]))))
        elif len(self.convolution_shape)==2:
            result = np.zeros((input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2]/self.stride_shape[0]))))

        #a list to save the outputs of convolution for each channel
        ch_conv_out = []
        #loop over batches
        for batch in range(input_tensor.shape[0]):

            # loop over different kernels
            for out_ch in range(self.weights.shape[0]):
                '''
                # mode: determines the output shape (using padding), e.g. 'same' means the output have the same
                 size as input (for a stride of 1), so pads in this way. 'valid' means no zero-padding.'''
                ch_conv_out = []

                # loop over each channel of the kernel and input
                for in_ch in range(self.weights.shape[1]):
                    ch_conv_out.append(signal.correlate(input_tensor[batch, in_ch], self.weights[out_ch, in_ch], mode='same', method='direct'))
                conv_plane = np.stack(ch_conv_out, axis=0)

                # after stacking the output of the correlation of each channel,
                # we should sum it over channels to get a 2d one-channel image
                conv_plane = conv_plane.sum(axis=0)

                #stride (down-sampling) implementation
                if len(self.convolution_shape)==3:
                    conv_plane = conv_plane[::self.stride_shape[0], ::self.stride_shape[1]]
                elif len(self.convolution_shape)==2:
                    conv_plane = conv_plane[::self.stride_shape[0]]

                # element-wise addition of bias for every kernel, each kernel has its own bias.
                result[batch, out_ch]= conv_plane + self.bias[out_ch]
        return result




    def backward(self, error_tensor):
        '''updates the parameters using the optimizer and returns the error tensor for the next layer'''
        # final result
        gradient_input = np.zeros_like(self.input_tensor)

        # we need new copy of weights and input in order not to over right the weights
        new_weights = np.copy(self.weights)

        # gradient of sth has always the same shape as it.
        self.gradient_weights = np.zeros_like(self.weights)

        if len(self.convolution_shape)==3:
            # gradient of sth has always the same shape as it.
            temp_gradient_weights = np.zeros((error_tensor.shape[0], self.weights.shape[0], self.weights.shape[1],
                                              self.weights.shape[2], self.weights.shape[3]))

            # padded_input = np.zeros((self.input_tensor.shape[0], self.input_tensor.shape[1], self.input_tensor.shape[2] +
            #      2 * (self.convolution_shape[1] // 2),
            #      self.input_tensor.shape[3] + 2 * (self.convolution_shape[2] // 2)))
            # pdb.set_trace()
            # padding for the width and height

            conv_plane_out = []
            for batch in range(self.input_tensor.shape[0]):
                # loop over different kernels (output channels)
                ch_conv_out = []
                for out_ch in range(self.input_tensor.shape[1]):
                    ch_conv_out.append(np.pad(self.input_tensor[batch, out_ch], ((self.convolution_shape[1]//2, self.convolution_shape[1]//2), (self.convolution_shape[2]//2, self.convolution_shape[2]//2)), mode='constant'))
                    if self.convolution_shape[2]%2 ==0:
                        ch_conv_out[out_ch] = ch_conv_out[out_ch][:,:-1]
                    if self.convolution_shape[1]%2 ==0:
                        ch_conv_out[out_ch] = ch_conv_out[out_ch][:-1,:]
                    # ch_conv_out[out_ch] = ch_conv_out[out_ch][::self.stride_shape[0], ::self.stride_shape[1]]
                # pdb.set_trace()
                conv_plane = np.stack(ch_conv_out, axis=0)
                conv_plane.tolist()
                conv_plane_out.append(conv_plane)
            padded_input = np.stack(conv_plane_out, axis=0)

                # padded_input[batch, out_ch] = np.pad(self.input_tensor[batch, out_ch],(self.convolution_shape[1]//2, self.convolution_shape[2]//2), mode='constant')

        # elif len(self.convolution_shape)==2:
        #     # gradient of sth has always the same shape as it.
        #     temp_gradient_weights = np.zeros((error_tensor.shape[0], self.weights.shape[0], self.weights.shape[1],
        #                                       self.weights.shape[2]))
        #
        #     # padded_input = np.zeros((self.input_tensor.shape[0], self.input_tensor.shape[1], self.input_tensor.shape[2] +
        #     #      2 * (self.convolution_shape[1] // 2)))
        #
        #     conv_plane_out = []
        #     # padding for the width and height
        #     for batch in range(self.input_tensor.shape[0]):
        #         ch_conv_out = []
        #
        #         # loop over different kernels (output channels)
        #         for out_ch in range(self.input_tensor.shape[1]):
        #             # padded_input[batch, out_ch] = np.pad(np.copy(self.input_tensor[batch, out_ch]),
        #             #                              (self.convolution_shape[1]//2), mode='constant')
        #             ch_conv_out.append(np.pad(self.input_tensor[batch, out_ch],  self.convolution_shape[1] // 2, mode='constant'))
        #             if self.convolution_shape[1]%2 ==0:
        #                 ch_conv_out[out_ch] = ch_conv_out[out_ch][:-1]
        #             # pdb.set_trace()
        #         conv_plane = np.stack(ch_conv_out, axis=0)
        #         conv_plane.tolist()
        #         conv_plane_out.append(conv_plane)
        #     padded_input = np.stack(conv_plane_out, axis=0)


        if len(self.convolution_shape)==3:
            '''gradient weight calculation'''
            #loop over batches
            for batch in range(error_tensor.shape[0]):
                # loop over different kernels (output channels)
                for out_ch in range(error_tensor.shape[1]):
                    # loop over input channels
                    for in_ch in range(self.input_tensor.shape[1]):
                        # pdb.set_trace()
                        temp = signal.resample(error_tensor[batch, out_ch], error_tensor[batch, out_ch].shape[0] * self.stride_shape[0], axis=0)
                        temp = signal.resample(temp, error_tensor[batch, out_ch].shape[1] * self.stride_shape[1], axis=1)
                        # slice it to match the correct shape if the last step of up-sampling was not full
                        temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]

                        temp_gradient_weights[batch, out_ch, in_ch] = signal.correlate(padded_input[batch, in_ch], temp, mode='valid')
            # pdb.set_trace()
            self.gradient_weights = temp_gradient_weights.sum(axis=0)



        # rearranging the weights according to the slide 49
        if len(self.convolution_shape)==3:
            new_weights = np.transpose(new_weights, (1,0,2,3))
        elif len(self.convolution_shape)==2:
            new_weights = np.transpose(new_weights, (1,0,2))


        #loop over batches
        for batch in range(error_tensor.shape[0]):
            # loop over different kernels (output channels)
            for out_ch in range(new_weights.shape[0]):

                # a list to save the outputs of convolution for each channel
                ch_conv_out = []

                # loop over each channel of the kernel and input
                for in_ch in range(new_weights.shape[1]):

                    # backward-stride (up-sampling) implementation
                    if len(self.convolution_shape) == 3:
                        temp = signal.resample(error_tensor[batch, in_ch], error_tensor[batch, in_ch].shape[0] * self.stride_shape[0], axis=0)
                        temp = signal.resample(temp, error_tensor[batch, in_ch].shape[1] * self.stride_shape[1], axis=1)
                        # slice it to match the correct shape if the last step of up-sampling was not full
                        temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]

                    elif len(self.convolution_shape) == 2:
                        temp = signal.resample(error_tensor[batch, in_ch], error_tensor[batch, in_ch].shape[0] * self.stride_shape[0], axis=0)
                        temp = temp[:self.input_tensor.shape[2]]

                    ch_conv_out.append(signal.convolve(temp, new_weights[out_ch, in_ch], mode='same', method='direct'))
                temp2 = np.stack(ch_conv_out, axis=0)

                # after stacking the output of the convolution of each channel,
                # we should sum it over channels to get a 2d one-channel image
                temp2 = temp2.sum(axis=0)

                # element-wise addition of bias for every kernel, each kernel has its own bias.
                gradient_input[batch, out_ch]= temp2

        #TODO: gradient weights calculation

        # for each kernel we have a scalar bias, so final bias is a vector
        if len(self.convolution_shape)==3:
            self.gradient_bias = np.sum(error_tensor, axis=(0,2,3))
        elif len(self.convolution_shape)==2:
            self.gradient_bias = np.sum(error_tensor, axis=(0,2))

        # Update
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return gradient_input


    def initialize(self, weights_initializer, bias_initializer):
        if len(self.convolution_shape) == 3:
            self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2]),
                                                          self.convolution_shape[0]*self.convolution_shape[1]* self.convolution_shape[2],
                                                          self.num_kernels*self.convolution_shape[1]* self.convolution_shape[2])
            self.bias = bias_initializer.initialize((self.num_kernels), 1, self.num_kernels)
            self.bias = self.bias[-1]

        elif len(self.convolution_shape) == 2:
            self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]),
                                                          self.convolution_shape[0]*self.convolution_shape[1],
                                                          self.num_kernels*self.convolution_shape[1])
            self.bias = bias_initializer.initialize((1, self.num_kernels), 1, self.num_kernels)
            self.bias = self.bias[-1]



    '''Properties'''

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
    def gradient_bias(self):
        return self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value
    @gradient_bias.deleter
    def gradient_bias(self):
        del self._gradient_bias


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