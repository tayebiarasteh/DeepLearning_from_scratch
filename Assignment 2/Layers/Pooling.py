import numpy as np
import skimage.measure
import pdb


class Pooling:
    def __init__(self, stride_shape=np.random.uniform(0,1,1)[0], pooling_shape=np.random.uniform(0,1,1)[0]):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape


    def forward(self, input_tensor):
        # pdb.set_trace()
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

        # loop over batches
        for i in range(input_tensor.shape[0]):
            # loop over each channel of the input
            for ii in range(input_tensor.shape[1]):

                #(normal without stride) maxpooling for each channel
                temp = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
                for iii in range(temp.shape[0]):
                    for iiii in range(temp.shape[1]):
                        temp[iii, iiii] = np.max(input_tensor[i,ii][iii:iii + self.pooling_shape[0], iiii:iiii + self.pooling_shape[1]])

                # stride implementation
                temp = temp[::self.stride_shape[0], ::self.stride_shape[1]]

                # removing the last column or row as mentioned above
                if offset_y:
                    temp = temp[:,:-1]
                if offset_x:
                    temp = temp[:-1,:]

                result[i, ii] = temp
        return result


    def backward(self, error_tensor):
        # pdb.set_trace()
        result = np.zeros_like(self.input_tensor)