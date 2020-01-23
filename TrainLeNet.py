from Layers import Helpers
from Models.LeNet import build
import NeuralNetwork
import matplotlib.pyplot as plt
import os.path

batch_size = 50
mnist = Helpers.MNISTData(batch_size)
mnist.show_random_training_image()

if os.path.isfile(os.path.join('trained', 'LeNet')):
    net = NeuralNetwork.load(os.path.join('trained', 'LeNet'), mnist)
else:
    net = build()
    net.data_layer = mnist

net.train(300)

NeuralNetwork.save(os.path.join('trained', 'LeNet'), net)

plt.figure('Loss function for training LeNet on the MNIST dataset')
plt.plot(net.loss, '-x')
plt.show()

data, labels = net.data_layer.get_test_set()

results = net.test(data)

accuracy = Helpers.calculate_accuracy(results, labels)
print('\nOn the MNIST dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')