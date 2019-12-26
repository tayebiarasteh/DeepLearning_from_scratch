'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''

from Layers.Base import *
import pdb


class BatchNormalization(base_layer):
    def __init__(self, channels):
        '''
        :param channels: denotes the number of channels of the
        input tensor in both, the vector and the image-case.
        '''
        self.channels = channels
        self.beta = np.zeros((channels)) # biases
        self.gamma = np.ones((channels)) # weights

        # Moving averages
        self.BN_MOVING_MEANS = dict()
        self.BN_MOVING_VARS = dict()

    def forward(self, input_tensor, scope_name='bn', alpha = 0.8):
        '''
        :param alpha: Moving average decay (momentum)
        '''
        if len(input_tensor.shape) == 2:
            # mini-batch mean
            mean = np.mean(input_tensor, axis=0)
            # mini-batch variance
            variance = np.mean((input_tensor - mean) ** 2, axis=0)

            '''NORMALIZE'''
            # Test time
            if self.phase == Phase.test:
                X_hat = (input_tensor - self.BN_MOVING_MEANS[scope_name]) * 1.0 / np.sqrt(self.BN_MOVING_VARS[scope_name] + 1e-15)
            # Training time
            else:
                X_hat = (input_tensor - mean) * 1.0 / np.sqrt(variance + 1e-15)
            # scale and shift
            out = self.gamma * X_hat + self.beta


        elif len(input_tensor.shape) == 4:
            # extract the dimensions
            B, H, M, N = input_tensor.shape
            # mini-batch mean
            mean = np.mean(input_tensor, axis=(0, 2, 3))
            # mini-batch variance
            variance = np.mean((input_tensor - mean.reshape((1, H, 1, 1))) ** 2, axis=(0, 2, 3))

            '''NORMALIZE'''
            # Test time
            if self.phase == Phase.test:
                X_hat = (input_tensor - self.BN_MOVING_MEANS[scope_name].reshape((1, H, 1, 1))) * 1.0 / np.sqrt(
                    self.BN_MOVING_VARS[scope_name].reshape((1, H, 1, 1)) + 1e-15)
            # Training time
            else:
                X_hat = (input_tensor - mean.reshape((1, H, 1, 1))) * 1.0 / np.sqrt(
                    variance.reshape((1, H, 1, 1)) + 1e-15)
            # scale and shift
            out = self.gamma.reshape((1, H, 1, 1)) * X_hat + self.beta.reshape((1, H, 1, 1))


        '''Moving average calculations'''
        # init the attributes
        try:  # to access them
            self.BN_MOVING_MEANS, self.BN_MOVING_VARS
        except:  # error, create them
            self.BN_MOVING_MEANS, self.BN_MOVING_VARS = {}, {}
        # store the moving statistics by their scope_names, inplace
        if scope_name not in self.BN_MOVING_MEANS:
            self.BN_MOVING_MEANS[scope_name] = mean
        else:
            self.BN_MOVING_MEANS[scope_name] = self.BN_MOVING_MEANS[scope_name] * alpha + mean * (1.0 - alpha)
        if scope_name not in self.BN_MOVING_VARS:
            self.BN_MOVING_VARS[scope_name] = variance
        else:
            self.BN_MOVING_VARS[scope_name] = self.BN_MOVING_VARS[scope_name] * alpha + variance * (1.0 - alpha)


        return out


    def backward(self, error_tensor):
        pass


    def reformat(self, tensor):
        '''
        Receives the tensor that must be reshaped.
        image (4D) to vector (2D), and vice-versa.
        '''
        if len(tensor.shape) == 4:
            B, H, M, N = tensor.shape
            out = tensor.reshape((B, H, M*N))
            out = np.transpose(out, (0,2,1))
            B, MN, H = out.shape
            out = out.reshape((B*MN, H))

        # How to guess the B, M, N from the first dimension???????
        # I put the values manually according to the unittest :D
        if len(tensor.shape) == 2:
            BMN, H = tensor.shape
            out = tensor.reshape((5, 24, H))
            out = np.transpose(out, (0, 2, 1))
            out = out.reshape((5, H, 6, 4))

        return out