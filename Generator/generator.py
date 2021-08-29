'''
Created on November 2019.
An image generator which returns the input of a neural network each time it gets called.
This input consists of a batch of images and its corresponding labels.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
'''

import os.path
import json
from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.transform import resize


class ImageGenerator:
    def __init__(self, file_path, json_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        '''
        :type image_size: tuple
        '''
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.path = (file_path, json_path)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.mirroring = mirroring
        self.rotation = rotation
        self.counter = 0    # shows the number of times next() has been called for each object of the class.
                            # if  self.counter =! 0 means that we have not created a new object.


    def next(self):
        '''This function creates a batch of images and corresponding labels and returns it.'''
        with open(self.path[1]) as data_file:
            label_file = json.load(data_file)
        all_images_indices = np.arange(len(label_file)) # indices of all the images in the dataset in a numpy array.

        images = [] # a batch (list) of images
        labels = [] # the corresponding labels

        if self.shuffle:
            np.random.shuffle(all_images_indices)

        '''If the last batch is smaller than the others, 
        complete that batch by reusing images from the beginning of your training data set:'''
        if (self.counter+1)*self.batch_size > len(label_file):
            offset = (self.counter+1)*self.batch_size - len(label_file)
            chosen_batch = all_images_indices[
                           self.counter * self.batch_size :len(label_file)]
            chosen_batch = np.append(chosen_batch, all_images_indices[0:offset])
            self.counter = -1   # at the end of the method with +1, it becomes zero and we basically reset our counter.
        else:
            chosen_batch = all_images_indices[self.counter*self.batch_size:(self.counter+1)*self.batch_size]

        for i in chosen_batch:
            images.append(np.load(os.path.join(self.path[0], str(i) + '.npy')))
            labels.append(label_file[str(i)])

        # Resizing
        for i, image in enumerate(images):
            images[i] = resize(image, self.image_size)

        # Augmentation
        for i, image in enumerate(images):
            images[i] = self.augment(image)

        # converting list to np array
        labels = np.asarray(labels)
        images = np.asarray(images)

        self.counter += 1
        output = (images, labels)
        return output


    def augment(self,img):
        '''This function takes a single image as an input and performs a random transformation
        (mirroring and/or rotation) on it and outputs the transformed image'''

        # mirroring (randomly)
        if self.mirroring:
            i = np.random.randint(0, 2, 1) # randomness
            if i[0] == 1: # 0: no | 1: yes
                img = np.fliplr(img)

        # rotation (randomly)
        if self.rotation:
            i = np.random.randint(0,4,1)
            i = i[0]
            img = np.rot90(img, i)

        return img


    def class_name(self, int_label):
        '''This function returns the class name for a specific input'''
        return self.class_dict[int_label]


    def show(self):
        '''In order to verify that the generator creates batches as required, this functions calls next to get a
        batch of images and labels and visualizes it.'''
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
