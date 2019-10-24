import os.path
import json
from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, json_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.path = (file_path, json_path)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.mirroring = mirroring
        self.rotation = rotation

        self.counter = 0    # shows the number of times next() has been called for each object of the class.

    def next(self):
        # This function creates a batch of images and corresponding labels and returns it.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        images = [] # a batch (list) of images
        labels = [] # the corresponding labels

        with open(self.path[1]) as data_file:
            label_file = json.load(data_file)
        all_images_indices = np.arange(len(label_file)) # indices of all the images in the dataset in a numpy array.
        if self.shuffle:
            np.random.shuffle(all_images_indices)

        '''If the last batch is smaller than the others, 
        complete that batch by reusing images from the beginning of your training data set:'''
        if (self.counter+1)*self.batch_size > len(label_file):
            offset = (self.counter+1)*self.batch_size - len(label_file)
            chosen_batch = all_images_indices[
                           self.counter * self.batch_size :len(label_file)]
            chosen_batch.append(all_images_indices[0:offset])
            self.counter = -1   # at the end of the method with +1, it becomes zero and we basically reset our counter.
        else:
            chosen_batch = all_images_indices[self.counter*self.batch_size:(self.counter+1)*self.batch_size]

        for i in chosen_batch:
            images.append(np.load(os.path.join(self.path[0], str(i) + '.npy')))
            labels.append(label_file[str(i)])

        for i, image in enumerate(images):
            images[i] = self.augment(image)

        self.counter += 1
        return (images, labels)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        # mirroring (randomly)
        if self.mirroring:
            i = np.random.randint(0, 2, 1) # randomness
            if i[0] == 1: # 0: no | 1: yes
                img = np.fliplr(img)

        # rotation (randomly)
        if self.rotation:
            angles = [0, 90, 180, 270]
            i = np.random.randint(0,4,1)
            i = i[0]
            i = angles[i]
            img = ndimage.rotate(img, i, reshape=False)

        return img

    def class_name(self, int_label):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[int_label]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        for i, image in enumerate(images):
            if self.batch_size > 3:
                n_rows = math.ceil(self.batch_size/3) # number of rows to plot for subplot
            else:
                n_rows = 1
            plt.subplot(n_rows, 3, i+1)
            plt.title(self.class_name(labels[i]))
            toPlot = plt.imshow(image)

            # hiding the axes text
            toPlot.axes.get_xaxis().set_visible(False)
            toPlot.axes.get_yaxis().set_visible(False)
        plt.show()





