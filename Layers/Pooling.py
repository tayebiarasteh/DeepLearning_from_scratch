'''
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/starasteh/
'''

import numpy as np
from Layers.Base import *


class Pooling(base_layer):
    def __init__(self, stride_shape=np.random.uniform(0,1,1)[0], pooling_shape=np.random.uniform(0,1,1)[0]):
        super().__init__()

        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.backward_result = 0
        self.input_tensor = 0


    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # we need these offsets because in the test it's not considering the last column or row if we have an odd number of them!!
        offset_y = input_tensor.shape[3] % self.stride_shape[1]
        if self.stride_shape[1]==1: # because dividing by 1 has remainder 0
            offset_y +=1
        offset_x = input_tensor.shape[2] % self.stride_shape[0]
        if self.stride_shape[0]==1: # because dividing by 1 has remainder 0
            offset_x +=1

        result = np.zeros((input_tensor.shape[0], input_tensor.shape[1],
                           int(np.ceil(input_tensor.shape[2] / self.stride_shape[0] - offset_x)),
                           int(np.ceil(input_tensor.shape[3] / self.stride_shape[1] - offset_y))))

        # locations of the maxima, to use it in the backward
        backward_result = np.zeros((input_tensor.shape[0], input_tensor.shape[1],
                           int(np.ceil(input_tensor.shape[2] / self.stride_shape[0] - offset_x)),
                           int(np.ceil(input_tensor.shape[3] / self.stride_shape[1] - offset_y))))

        # loop over batches
        for batch in range(input_tensor.shape[0]):
            # loop over each channel of the input
            for ch in range(input_tensor.shape[1]):

                #(normal without stride) maxpooling for each channel
                pool_plane = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

                # for storing the locations of the maxima
                loc_plane = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

                for i in range(pool_plane.shape[0]):
                    for ii in range(pool_plane.shape[1]):
                        #maxpooling
                        pool_plane[i, ii] = np.max(input_tensor[batch, ch][i:i + self.pooling_shape[0], ii:ii + self.pooling_shape[1]])

                        # storing the locations of maximas
                        loc_plane[i, ii] = np.argmax(input_tensor[batch, ch][i:i + self.pooling_shape[0], ii:ii + self.pooling_shape[1]])

                        # fine tuning the locations to be general in the image dimension of the input
                        if i == pool_plane.shape[0]-1 and ii != pool_plane.shape[1]-1:
                            loc_plane[i, ii] += (pool_plane.shape[0]-1) * pool_plane.shape[1]
                        elif ii == pool_plane.shape[1]-1 and i != pool_plane.shape[0]-1:
                            if loc_plane[i, ii] == 1.0:
                                loc_plane[i, ii] += pool_plane.shape[1]
                            loc_plane[i, ii] += i * pool_plane.shape[1]
                        elif i == pool_plane.shape[0]-1 and ii == pool_plane.shape[1]-1:
                            loc_plane[i, ii] = pool_plane.shape[0] * pool_plane.shape[1] -1
                        else:
                            if loc_plane[i, ii]>1.0:
                                loc_plane[i, ii] += pool_plane.shape[1] -2
                            loc_plane[i, ii] += ii + pool_plane.shape[1]*i

                # stride implementation
                pool_plane = pool_plane[::self.stride_shape[0], ::self.stride_shape[1]]
                loc_plane = loc_plane[::self.stride_shape[0], ::self.stride_shape[1]]

                # removing the last column or row as mentioned above
                if offset_y==1:
                    pool_plane = pool_plane[:,:-1]
                    loc_plane = loc_plane[:,:-1]
                if offset_x==1:
                    pool_plane = pool_plane[:-1,:]
                    loc_plane = loc_plane[:-1,:]

                result[batch, ch] = pool_plane
                backward_result[batch, ch] = loc_plane
        self.backward_result = backward_result.astype(int)

        # layout preservation
        if self.stride_shape == (1,1):
            return input_tensor

        return result


    def backward(self, error_tensor):
        '''returns the error tensor for the next layer.'''
        result = np.zeros_like(self.input_tensor)

        # loop over batches
        for batch in range(self.input_tensor.shape[0]):
            # loop over each channel of the input
            for ch in range(self.input_tensor.shape[1]):

                # nested-loop over locations of the maxima
                for i, location_vec in enumerate(self.backward_result[batch, ch]):
                    for ii, location in enumerate(location_vec):

                        # gives the coordinate-like indices
                        coord_x, coord_y = np.unravel_index(location, self.input_tensor[0, 0].shape)

                        '''backward maxpooling'''
                        # if self.stride_shape < self.pooling_shape
                        # then we may have some duplicates. So we sum up the values which refer to the same location.
                        if result[batch, ch, coord_x, coord_y] != 0.0:
                            result[batch, ch, coord_x, coord_y] += error_tensor[batch, ch, i, ii]
                        else:
                            result[batch, ch, coord_x, coord_y] = error_tensor[ batch, ch, i, ii]
        return result