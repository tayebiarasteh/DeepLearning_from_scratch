'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/starasteh/
'''

from Layers.Base import *
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(base_layer):
    def __init__(self, channels):
        '''
        :param channels: denotes the number of channels of the
        input tensor in both, the vector and the image-case.
        '''
        self.channels = channels
        self.bias = np.zeros((channels)) # beta
        self.weights = np.ones((channels)) # gamma
        self._gradient_bias = None
        self._gradient_weights = None
        self._optimizer = None #weight optimizer
        self._bias_optimizer = None
        self.mean = 0
        self.variance = 1
        self.test_mean = 0
        self.test_variance = 1


    def forward(self, input_tensor, alpha = 0.8):
        '''
        :param alpha: Moving average decay (momentum)
        '''
        epsilon = 1e-15
        self.input_tensor = input_tensor

        if len(input_tensor.shape) == 2:
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)

            if (self.phase == Phase.train):
                # mini-batch mean
                new_mean = np.mean(input_tensor, axis=0)
                # mini-batch variance
                new_variance = np.var((input_tensor), axis=0)

                '''Moving average calculations for test time'''
                self.test_mean = alpha * self.mean + (1 - alpha) * new_mean
                self.test_variance = alpha * self.variance + (1 - alpha) * new_variance

                self.mean = new_mean
                self.variance = new_variance

                # NORMALIZE
                X_hat = (input_tensor - self.mean) / np.sqrt(self.variance + epsilon)

            # NORMALIZE test time
            if self.phase == Phase.test:
                X_hat = (input_tensor - self.test_mean) / np.sqrt(self.test_variance + epsilon)

            self.X_hat = X_hat
            out = self.weights * X_hat + self.bias


        elif len(input_tensor.shape) == 4:
            # extract the dimensions
            B, H, M, N = input_tensor.shape
            self.mean = np.mean(input_tensor, axis=(0, 2, 3))
            self.variance = np.var((input_tensor), axis=(0, 2, 3))

            if (self.phase == Phase.train):
                # mini-batch mean
                new_mean = np.mean(input_tensor, axis=(0, 2, 3))
                # mini-batch variance
                new_variance = np.var(input_tensor, axis=(0, 2, 3))

                '''Moving average calculations for test time'''
                self.test_mean = alpha * self.mean.reshape((1, H, 1, 1)) + (1 - alpha) * new_mean.reshape((1, H, 1, 1))
                self.test_variance = alpha * self.variance.reshape((1, H, 1, 1)) + (1 - alpha) * new_variance.reshape((1, H, 1, 1))

                self.mean = new_mean
                self.variance = new_variance

                # NORMALIZE
                X_hat = (input_tensor - self.mean.reshape((1, H, 1, 1))) / np.sqrt(self.variance.reshape((1, H, 1, 1)) + epsilon)

            # NORMALIZE test time
            if self.phase == Phase.test:
                X_hat = (input_tensor - self.test_mean.reshape((1, H, 1, 1))) / np.sqrt(self.test_variance.reshape((1, H, 1, 1)) + epsilon)

            self.X_hat = X_hat
            out = self.weights.reshape((1, H, 1, 1)) * X_hat + self.bias.reshape((1, H, 1, 1))

        return out


    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:
            out = compute_bn_gradients(self.reformat(error_tensor), self.reformat(self.input_tensor),
                                       self.weights, self.mean, self.variance, 1e-15)
            out = self.reformat(out)
            self.gradient_weights = np.sum(error_tensor * self.X_hat, axis=(0,2,3))
            self.gradient_bias = np.sum(error_tensor, axis=(0,2,3))

        elif len(error_tensor.shape) == 2:
            out = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.variance, 1e-15)
            self.gradient_weights = np.sum(error_tensor * self.X_hat, axis=0)
            self.gradient_bias = np.sum(error_tensor, axis=0)

        '''Update with optimizers'''
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return out


    def reformat(self, tensor):
        '''
        Receives the tensor that must be reshaped.
        image (4D) to vector (2D), and vice-versa.
        '''
        out = np.zeros_like(tensor)
        if len(tensor.shape) == 4:
            B, H, M, N = tensor.shape
            out = tensor.reshape((B, H, M*N))
            out = np.transpose(out, (0,2,1))
            B, MN, H = out.shape
            out = out.reshape((B*MN, H))

        # How to guess the B, M, N from the first dimension??
        if len(tensor.shape) == 2:
            try:
                B, H, M, N = self.input_shape
            except:
                B, H, M, N = self.input_tensor.shape
            out = tensor.reshape((B, M * N, H))
            out = np.transpose(out, (0, 2, 1))
            out = out.reshape((B, H, M, N))

        return out


    def initialize(self, weights_initializer, bias_initializer):
        self.bias = np.zeros((self.channels)) # beta
        self.weights = np.ones((self.channels)) # gamma



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
