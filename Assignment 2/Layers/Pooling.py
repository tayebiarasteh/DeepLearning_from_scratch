import numpy as np
import pdb


class Pooling:
    def __init__(self, stride_shape=np.random.uniform(0,1,1)[0], pooling_shape=np.random.uniform(0,1,1)[0]):
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

        # to use it in the backward
        backward_result = np.zeros((input_tensor.shape[0], input_tensor.shape[1],
                           int(np.ceil(input_tensor.shape[2] / self.stride_shape[0] - offset_x)),
                           int(np.ceil(input_tensor.shape[3] / self.stride_shape[1] - offset_y))))

        # loop over batches
        for i in range(input_tensor.shape[0]):
            # loop over each channel of the input
            for ii in range(input_tensor.shape[1]):

                #(normal without stride) maxpooling for each channel
                temp = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

                # for storing the locations of the maxima
                temp2 = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

                for iii in range(temp.shape[0]):
                    for iiii in range(temp.shape[1]):
                        temp[iii, iiii] = np.max(input_tensor[i,ii][iii:iii + self.pooling_shape[0], iiii:iiii + self.pooling_shape[1]])

                        # storing the locations of maximas
                        temp2[iii, iiii] = np.argmax(input_tensor[i,ii][iii:iii + self.pooling_shape[0], iiii:iiii + self.pooling_shape[1]])
                        if iii == temp.shape[0]-1 and iiii != temp.shape[1]-1:
                            temp2[iii, iiii] += (temp.shape[0]-1) * temp.shape[1]
                        elif iiii == temp.shape[1]-1 and iii != temp.shape[0]-1:
                            if temp2[iii, iiii] == 1.0:
                                temp2[iii, iiii] += temp.shape[1]
                            temp2[iii, iiii] += iii * temp.shape[1]
                        elif iii == temp.shape[0]-1 and iiii == temp.shape[1]-1:
                            temp2[iii, iiii] = temp.shape[0] * temp.shape[1] -1
                        else:
                            if temp2[iii, iiii]>1.0:
                                temp2[iii, iiii] += temp.shape[1] -2
                            temp2[iii, iiii] += iiii + temp.shape[1]*iii

                # stride implementation
                temp = temp[::self.stride_shape[0], ::self.stride_shape[1]]
                temp2 = temp2[::self.stride_shape[0], ::self.stride_shape[1]]

                # removing the last column or row as mentioned above
                if offset_y==1:
                    temp = temp[:,:-1]
                    temp2 = temp2[:,:-1]
                if offset_x==1:
                    temp = temp[:-1,:]
                    temp2 = temp2[:-1,:]

                result[i, ii] = temp
                backward_result[i, ii] = temp2
        self.backward_result = backward_result.astype(int)

        # layout preservation
        if self.stride_shape == (1,1):
            return input_tensor

        return result


    def backward(self, error_tensor):
        '''returns the error tensor for the next layer.'''
        # pdb.set_trace()
        result = np.zeros_like(self.input_tensor)

        # loop over batches
        for i in range(self.input_tensor.shape[0]):
            # loop over each channel of the input
            for ii in range(self.input_tensor.shape[1]):
                # pdb.set_trace()
                for iii, item1 in enumerate(self.backward_result[i,ii]):
                    for iiii, item2 in enumerate(item1):

                        # gives the coordinate-like indices
                        temp1, temp2 = np.unravel_index(item2, self.input_tensor[0, 0].shape)

                        # backward maxpooling
                        result[i,ii, temp1, temp2] = error_tensor[i,ii,iii,iiii]
                        # if self.stride_shape < self.pooling_shape:
                        #     np.sum(result[i,ii], axis=0)
        # pdb.set_trace()
        return result