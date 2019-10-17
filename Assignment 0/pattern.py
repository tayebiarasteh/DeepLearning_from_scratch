import numpy as np
import matplotlib.pyplot as plt


class Checker():

    def __init__(self, resolution, tile_size):
        '''
        :param resolution: an integer that defines the number of pixels in each dimension
        :param tile_size: an integer that defines the number of pixel an individual tile has in each dimension.
        '''
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = []

    def draw(self):
        assert self.resolution % 2*(self.tile_size) == 0
        # number of the tiles
        half_tiles_num = int(self.resolution/(2*self.tile_size))
        self.output = np.array([[0,1],[1,0]])
        self.output = np.tile(self.output, (half_tiles_num, half_tiles_num))
        return self.output

    def show(self):
        plt.imshow(self.output)
        plt.show()







class Spectrum():

    def __init__(self):
        pass




class Circle():

    def __init__(self):
        pass
