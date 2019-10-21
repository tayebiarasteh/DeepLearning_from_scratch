import numpy as np
import matplotlib.pyplot as plt


class Checker:
    """
    a checkerboard pattern with adaptable tile size and resolution.
    """

    def __init__(self, resolution, tile_size):
        '''
        :param resolution: an integer that defines the number of pixels in each dimension
        :param tile_size: an integer that defines the number of pixel an individual tile has in each dimension.
        '''
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.array([])

    def draw(self):
        assert self.resolution % 2*(self.tile_size) == 0
        # half of the number of the tiles
        half_tiles_num = int(self.resolution/(2*self.tile_size))
        self.output = np.array([[0,1],[1,0.0]])
        self.output = np.tile(self.output, (half_tiles_num, half_tiles_num))
        return self.output

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()



class Spectrum:
    '''
    an RGB color spectrum.
    '''

    def __init__(self, resolution):
        """
        :type resolution: int
        """
        self.resolution = resolution
        self.output = np.zeros([self.resolution, self.resolution, 3])  # init the array

    def draw(self):
        # Red channel
        self.output[:, :, 0] = np.linspace(0.0, 1.0, self.resolution)
        # Blue channel
        self.output[:, :, 2] = np.linspace(1.0, 0.0, self.resolution)
        # Green channel
        self.output[:, :, 1] = np.linspace(0.0, 1.0, self.resolution)
        self.output[:, :, 1] = self.output[:, :, 1].T #transposing the green matrix
        return self.output

    def show(self):
        plt.imshow(self.output)
        plt.show()



class Circle:
    '''
    a binary circle with a given radius at a specified position in the image.
    '''

    def __init__(self, resolution, radius, position):
        """
        :type position: tuple, describes the position of the circle center in the image.
        :type radius: integer, describes the radius of the circle.
        :type resolution: integer
        """
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution))
        y, x = np.ogrid[0:self.resolution, 0:self.resolution]
        mask = ((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2 < self.radius ** 2)
        self.output[mask] = 1.0
        return self.output


    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
