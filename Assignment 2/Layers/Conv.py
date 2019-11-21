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
        # pdb.set_trace()
        if len(self.convolution_shape)==3:
            result = np.zeros((input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2]/self.stride_shape[0])),
                               int(np.ceil(input_tensor.shape[3]/self.stride_shape[1]))))
        elif len(self.convolution_shape)==2:
            result = np.zeros((input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2]/self.stride_shape[0]))))

        #loop over batches
        for i in range(input_tensor.shape[0]):

            # loop over different kernels
            for ii in range(self.weights.shape[0]):
                '''
                # correlation gives an output with channels, we should sum it over channels to get a 2d one-channel image
                # mode: determines the output shape (using padding), e.g. 'same' means the
                    output have the same size as input (for a stride of 1), so pads in this way.
                    'valid' means no zero-padding.
                # this function works with "channels first"'''
                temp1 = signal.correlate(input_tensor[i], self.weights[ii], mode='same', method='direct').sum(axis=0)

                #stride implementation
                if len(self.convolution_shape)==3:
                    temp1 = temp1[::self.stride_shape[0],::self.stride_shape[1]]
                elif len(self.convolution_shape)==2:
                    temp1 = temp1[::self.stride_shape[0]]

                result[i,ii]= temp1

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