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
        self.bias = 0
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
        for i in range(input_tensor.shape[0]):

            # loop over different kernels
            for ii in range(self.weights.shape[0]):
                '''
                # mode: determines the output shape (using padding), e.g. 'same' means the output have the same
                 size as input (for a stride of 1), so pads in this way. 'valid' means no zero-padding.'''

                # loop over each channel of the kernel and input
                for iii in range(self.weights.shape[1]):
                    ch_conv_out.append(signal.correlate(input_tensor[i, iii], self.weights[ii, iii], mode='same', method='direct'))
                temp2 = np.stack(ch_conv_out, axis=0)

                # after stacking the output of the correlation of each channel,
                # we should sum it over channels to get a 2d one-channel image
                temp2 = temp2.sum(axis=0)

                #stride (down-sampling) implementation
                if len(self.convolution_shape)==3:
                    temp2 = temp2[::self.stride_shape[0], ::self.stride_shape[1]]
                elif len(self.convolution_shape)==2:
                    temp2 = temp2[::self.stride_shape[0]]

                #each kernel has its own bias
                result[i,ii]= temp2 + self.bias
        return result




    def backward(self, error_tensor):
        '''updates the parameters using the optimizer and returns the error tensor for the next layer'''
        # pdb.set_trace()
        result = np.zeros_like(self.input_tensor)
        # rearranging the weights according to the slide 49
        if len(self.convolution_shape)==3:
            self.weights = np.transpose(self.weights, (1,0,2,3))
        elif len(self.convolution_shape)==2:
            self.weights = np.transpose(self.weights, (1,0,2))

        # a list to save the outputs of convolution for each channel
        ch_conv_out = []
        #loop over batches
        for i in range(error_tensor.shape[0]):
            # loop over different kernels
            for ii in range(self.weights.shape[0]):
                # loop over each channel of the kernel and input
                for iii in range(self.weights.shape[1]):

                    # backward-stride (up-sampling) implementation
                    if len(self.convolution_shape) == 3:
                        temp = signal.resample(error_tensor[i,iii], error_tensor[i,iii].shape[0] * self.stride_shape[0], axis=0)
                        temp = signal.resample(temp, error_tensor[i,iii].shape[1] * self.stride_shape[1], axis=1)
                        # slice it to match the correct shape if the last step of upsampling was not full
                        temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]

                    elif len(self.convolution_shape) == 2:
                        temp = signal.resample(error_tensor[i,iii], error_tensor[i,iii].shape[0] * self.stride_shape[0], axis=0)
                        temp = temp[:self.input_tensor.shape[2]]

                    ch_conv_out.append(signal.convolve(temp, self.weights[ii, iii], mode='same', method='direct'))
                temp2 = np.stack(ch_conv_out, axis=0)

                # after stacking the output of the convolution of each channel,
                # we should sum it over channels to get a 2d one-channel image
                temp2 = temp2.sum(axis=0)

                #each kernel has its own bias
                result[i,ii]= temp2 + self.bias

        # self.gradient_weights = np.matmul(self.input_tensor.T, error_tensor)

        return result


    # def initialize(self, weights_initializer, bias_initializer):
    #     self.weights = weights_initializer.initialize((self.output_size, self.input_size), self.input_size, self.output_size)
    #     self.bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
    #     # self.weights = np.vstack((self.weights, self.bias))


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
        self._bias_optimizer = value
    @optimizer.deleter
    def optimizer(self):
        del self._optimizer