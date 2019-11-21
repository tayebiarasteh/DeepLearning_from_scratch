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
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None


    def forward(self, input_tensor):
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

                #stride implementation
                if len(self.convolution_shape)==3:
                    temp2 = temp2[::self.stride_shape[0],::self.stride_shape[1]]
                elif len(self.convolution_shape)==2:
                    temp2 = temp2[::self.stride_shape[0]]

                #each kernel has its own bias
                result[i,ii]= temp2 + self.bias
        return result




    def backward(self, error_tensor):
        pass



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