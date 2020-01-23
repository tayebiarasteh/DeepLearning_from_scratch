import unittest
from Layers import *
from Optimization import *
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import os
import datetime
import pdb

RNN_TEST = False

class TestFullyConnected(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_backward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([ self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = FullyConnected.FullyConnected(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)

class TestReLU(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.ones([self.batch_size, self.input_size])
        self.input_tensor[0:self.half_batch_size,:] -= 2

        self.label_tensor = np.zeros([self.batch_size, self.input_size])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_forward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 1

        layer = ReLU.ReLU()
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_backward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 2

        layer = ReLU.ReLU()
        layer.forward(self.input_tensor)
        output_tensor = layer.backward(self.input_tensor*2)
        self.assertEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        input_tensor *= 2.
        input_tensor -= 1.
        layers = list()
        layers.append(ReLU.ReLU())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestTanH(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        self.input_tensor *= 2.
        self.input_tensor -= 1.

        self.label_tensor = np.zeros([self.input_size, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_forward(self):
        expected_tensor = 1 - 2 / (np.exp(2*self.input_tensor) + 1)

        layer = TanH.TanH()
        output_tensor = layer.forward(self.input_tensor)
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_range(self):
        layer = TanH.TanH()
        output_tensor = layer.forward(self.input_tensor*2)

        out_max = np.max(output_tensor)
        out_min = np.min(output_tensor)

        self.assertLessEqual(out_max, 1.)
        self.assertGreaterEqual(out_min, -1.)

    def test_gradient(self):
        layers = list()
        layers.append(TanH.TanH())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        self.input_tensor *= 2.
        self.input_tensor -= 1.

        self.label_tensor = np.zeros([self.input_size, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_forward(self):
        expected_tensor = 0.5 * (1. + np.tanh(self.input_tensor / 2.))

        layer = Sigmoid.Sigmoid()
        output_tensor = layer.forward(self.input_tensor)
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_range(self):
        layer = Sigmoid.Sigmoid()
        output_tensor = layer.forward(self.input_tensor*2)

        out_max = np.max(output_tensor)
        out_min = np.min(output_tensor)

        self.assertLessEqual(out_max, 1.)
        self.assertGreaterEqual(out_min, 0.)

    def test_gradient(self):
        layers = list()
        layers.append(Sigmoid.Sigmoid())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestSoftMax(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_shift(self):
        input_tensor = np.zeros([self.batch_size, self.categories]) + 10000.
        layer = SoftMax.SoftMax()
        pred = layer.forward(input_tensor)
        self.assertFalse(np.isnan(np.sum(pred)))

    def test_forward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertLess(loss, 1e-10)

    def test_backward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()
        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        self.assertAlmostEqual(np.sum(error), 0)

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertAlmostEqual(float(loss), 12)

    def test_regression_backward_high_loss_w_CrossEntropy(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertGreaterEqual(element, 1/3)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertAlmostEqual(element, -1)

    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)

        # just see if it's bigger then zero
        self.assertGreater(float(loss), 0.)


    def test_regression_backward(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertLessEqual(element, 0)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertGreaterEqual(element, 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(SoftMax.SoftMax())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_predict(self):
        input_tensor = np.arange(self.categories * self.batch_size)
        input_tensor = input_tensor / 100.
        input_tensor = input_tensor.reshape((self.categories, self.batch_size))
        # print(input_tensor)
        layer = SoftMax.SoftMax()
        prediction = layer.forward(input_tensor.T)
        # print(prediction)
        expected_values = np.array([[0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724,
                                     0.21732724, 0.21732724],
                                    [0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387,
                                     0.23779387, 0.23779387],
                                    [0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794,
                                     0.26018794, 0.26018794],
                                    [0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095,
                                     0.28469095, 0.28469095]])
        # print(expected_values)
        # print(prediction)
        np.testing.assert_almost_equal(expected_values, prediction.T)

class TestCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(Loss.CrossEntropyLoss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_zero_loss(self):
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(self.label_tensor, self.label_tensor)
        self.assertAlmostEqual(loss, 0)

    def test_high_loss(self):
        label_tensor = np.zeros((self.batch_size, self.categories))
        label_tensor[:, 2] = 1
        input_tensor = np.zeros_like(label_tensor)
        input_tensor[:, 1] = 1
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(input_tensor, label_tensor)
        self.assertAlmostEqual(loss, 324.3928805, places = 4)

class TestOptimizers(unittest.TestCase):

    def test_sgd(self):
        optimizer = Optimizers.Sgd(1.)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.]))

    def test_sgd_with_momentum(self):
        optimizer = Optimizers.SgdWithMomentum(1., 0.9)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.9]))

    def test_adam(self):
        optimizer = Optimizers.Adam(1., 0.01, 0.02)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-0.99999998000000034]))


class TestInitializers(unittest.TestCase):
    class DummyLayer:
        def __init__(self, input_size, output_size):
            self.weights = []
            self.shape = (output_size, input_size)

        def initialize(self, initializer):
            self.weights = initializer.initialize(self.shape, self.shape[1], self.shape[0])

    def setUp(self):
        self.batch_size = 9
        self.input_size = 400
        self.output_size = 400
        self.num_kernels = 20
        self.num_channels = 20
        self.kernelsize_x = 41
        self.kernelsize_y = 41

    def _performInitialization(self, initializer):
        np.random.seed(1337)
        layer = TestInitializers.DummyLayer(self.input_size, self.output_size)
        layer.initialize(initializer)
        weights_after_init = layer.weights.copy()
        return layer.shape, weights_after_init

    def test_uniform_shape(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.UniformRandom())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_uniform_distribution(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.UniformRandom())

        p_value = stats.kstest(weights_after_init.flat, 'uniform', args=(0, 1)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_xavier_shape(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.Xavier())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_xavier_distribution(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.Xavier())

        scale = np.sqrt(2) / np.sqrt(self.input_size + self.output_size)
        p_value = stats.kstest(weights_after_init.flat, 'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_he_shape(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.He())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_he_distribution(self):
        weights_before_init, weights_after_init = self._performInitialization(Initializers.He())

        scale = np.sqrt(2) / np.sqrt(self.input_size)
        p_value = stats.kstest(weights_after_init.flat, 'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)


class TestFlatten(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.input_shape = (3, 4, 11)
        self.input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=np.float)
        self.input_tensor = self.input_tensor.reshape(self.batch_size, *self.input_shape)

    def test_flatten_forward(self):
        flatten = Flatten.Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        input_vector = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=np.float)
        input_vector = input_vector.reshape(self.batch_size, np.prod(self.input_shape))
        self.assertLessEqual(np.sum(np.abs(output_tensor-input_vector)), 1e-9)

    def test_flatten_backward(self):
        flatten = Flatten.Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        backward_tensor = flatten.backward(output_tensor)
        self.assertLessEqual(np.sum(np.abs(self.input_tensor - backward_tensor)), 1e-9)


class TestConv(unittest.TestCase):
    plot = False
    directory = 'plots/'

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros((1, 3, 3, 3))
            weights[0, 1, 1, 1] = 1
            return weights

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (3, 10, 14)
        self.input_size = 14 * 10 * 3
        self.uneven_input_shape = (3, 11, 15)
        self.uneven_input_size = 15 * 11 * 3
        self.spatial_input_shape = np.prod(self.input_shape[1:])
        self.kernel_shape = (3, 5, 8)
        self.num_kernels = 4
        self.hidden_channels = 3

        self.categories = 5
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_size(self):
        conv = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, *self.input_shape[1:]))

    def test_forward_size_stride(self):
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, 4, 7))

    def test_forward_size_stride_uneven_image(self):
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels + 1)
        input_tensor = np.array(range(int(np.prod(self.uneven_input_shape) * (self.batch_size + 1))), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size + 1, *self.uneven_input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, ( self.batch_size+1, self.num_kernels+1, 4, 8))

    def test_forward(self):
        np.random.seed(1337)
        conv = Conv.Conv((1, 1), (1, 3, 3), 1)
        conv.weights = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        input_tensor = np.random.random((1, 1, 10, 14))
        expected_output = gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertAlmostEqual(difference, 0., places=1)

    def test_forward_multi_channel(self):
        np.random.seed(1337)
        maps_in = 2
        bias = 1
        conv = Conv.Conv((1, 1), (maps_in, 3, 3), 1)
        filter = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.weights = np.repeat(filter[None, ...], maps_in, axis=1)
        conv.bias = np.array([bias])
        input_tensor = np.random.random((1, maps_in, 10, 14))
        expected_output = bias
        for map_i in range(maps_in):
            expected_output = expected_output + gaussian_filter(input_tensor[0, map_i, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor) / maps_in)
        self.assertAlmostEqual(difference, 0., places=1)

    def test_forward_fully_connected_channels(self):
        np.random.seed(1337)
        conv = Conv.Conv((1, 1), (3, 3, 3), 1)
        conv.weights = (1. / 15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        tensor = np.random.random((1, 1, 10, 14))
        input_tensor = np.zeros((1, 3 , 10, 14))
        input_tensor[:,0] = tensor.copy()
        input_tensor[:,1] = tensor.copy()
        input_tensor[:,2] = tensor.copy()
        expected_output = 3 * gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertLess(difference, 0.2)

    def test_1D_forward_size(self):
        conv = Conv.Conv([2], (3, 3), self.num_kernels)
        input_tensor = np.array(range(3 * 15 * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, 3, 15)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape,  (self.batch_size,self.num_kernels, 8))

    def test_backward_size(self):
        conv = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape))

    def test_backward_size_stride(self):
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape))

    def test_1D_backward_size(self):
        conv = Conv.Conv([2], (3, 3), self.num_kernels)
        input_tensor = np.array(range(45 * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, 3, 15)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, 3, 15))


    def test_1x1_convolution(self):
        conv = Conv.Conv((1, 1), (3, 1, 1), self.num_kernels)
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, *self.input_shape[1:]))
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape))


    def test_layout_preservation(self):
        conv = Conv.Conv((1, 1), (3, 3, 3), 1)
        conv.initialize(self.TestInitializer(), Initializers.Constant(0.0))
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(np.squeeze(output_tensor)-input_tensor[:,1,:,:])), 0.)

    def test_gradient(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(FullyConnected.FullyConnected(35 * self.hidden_channels, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 5e-2)

    def test_gradient_weights(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(FullyConnected.FullyConnected(35*self.hidden_channels, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)

        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights_strided(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((2, 2), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(FullyConnected.FullyConnected(12*self.hidden_channels, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)

        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_bias(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(FullyConnected.FullyConnected(35 * self.hidden_channels, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, True)

        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_stride(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 14)))
        layers = list()
        layers.append(Conv.Conv((1, 2), (3, 3, 3), 1))
        layers.append(Flatten.Flatten())
        layers.append(FullyConnected.FullyConnected(35, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_update(self):
        input_tensor = np.abs(np.random.random((self.batch_size, *self.input_shape)))
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels)
        conv.optimizer = Optimizers.SgdWithMomentum(1, 0.9)
        conv.initialize(Initializers.He(), Initializers.Constant(0.1))
        for _ in range(10):
            output_tensor = conv.forward(input_tensor)
            error_tensor = np.zeros_like(output_tensor)
            error_tensor -= output_tensor
            conv.backward(error_tensor)
            new_output_tensor = conv.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_initialization(self):
        conv = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        init = TestConv.TestInitializer()
        conv.initialize(init, Initializers.Constant(0.1))
        self.assertEqual(init.fan_in, np.prod(self.kernel_shape))
        self.assertEqual(init.fan_out, np.prod(self.kernel_shape[1:]) * self.num_kernels)


class TestPooling(unittest.TestCase):
    plot = False
    directory = 'plots/'

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (2, 4, 7)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(1337)
        self.input_tensor = np.abs(np.random.random((self.batch_size, *self.input_shape)))

        self.categories = 5
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(Flatten.Flatten())
        self.layers.append(None)
        self.layers.append(L2Loss())
        self.plot_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))

    def test_shape(self):
        layer = Pooling.Pooling((2, 2), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 3])
        self.assertEqual(np.abs(np.sum(np.array(result.shape) - expected_shape)), 0)

    def test_overlapping_shape(self):
        layer = Pooling.Pooling((2, 1), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 6])
        self.assertEqual(np.abs(np.sum(np.array(result.shape) - expected_shape)), 0)

    def test_subsampling_shape(self):
        layer = Pooling.Pooling((3, 2), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 1, 3])
        self.assertEqual(np.abs(np.sum(np.array(result.shape) - expected_shape)), 0)

    def test_gradient_stride(self):
        self.layers[0] = Pooling.Pooling((2, 2), (2, 2))
        self.layers[2] = FullyConnected.FullyConnected(12, self.categories)

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_overlapping_stride(self):
        self.layers[0] = Pooling.Pooling((2, 1), (2, 2))
        self.layers[2] = FullyConnected.FullyConnected(24, self.categories)
        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_subsampling_stride(self):
        self.layers[0] = Pooling.Pooling((3, 2), (2, 2))
        self.layers[2] = FullyConnected.FullyConnected(6, self.categories)
        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_layout_preservation(self):
        pool = Pooling.Pooling((1, 1), (1, 1))
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = pool.forward(input_tensor)

        self.assertAlmostEqual(np.sum(np.abs(output_tensor-input_tensor)), 0.)

    def test_expected_output_valid_edgecase(self):
        input_shape = (1, 3, 3)
        pool = Pooling.Pooling((2, 2), (2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)

        result = pool.forward(input_tensor)
        expected_result = np.array([[4], [13]]).T
        self.assertEqual(np.abs(np.sum(result - expected_result)), 0)

    def test_expected_output(self):
        input_shape = (1, 4, 4)
        pool = Pooling.Pooling((2, 2), (2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)

        result = pool.forward(input_tensor)
        expected_result = np.array([[[[ 5.,  7.],[13., 15.]]],[[[21., 23.],[29., 31.]]]]).T
        self.assertEqual(np.abs(np.sum(result - expected_result)), 0)


class TestConstraints(unittest.TestCase):

    def setUp(self):
        self.delta = 0.1
        self.regularizer_strength = 1337
        self.shape = (4, 5)

    def test_L1(self):
        regularizer = Constraints.L1_Regularizer(self.regularizer_strength)

        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1
        weights_tensor = regularizer.calculate_gradient(weights_tensor)

        expected = np.ones(self.shape) * self.regularizer_strength
        expected[1:3, 2:4] *= -1

        difference = np.sum(np.abs(weights_tensor - expected))
        self.assertLessEqual(difference, 1e-10)

    def test_L1_norm(self):
        regularizer = Constraints.L1_Regularizer(self.regularizer_strength)

        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1
        norm = regularizer.norm(weights_tensor)
        self.assertAlmostEqual(norm, 20*self.regularizer_strength)

    def test_L2(self):
        regularizer = Constraints.L2_Regularizer(self.regularizer_strength)

        weights_tensor = np.ones(self.shape)
        weights_tensor = regularizer.calculate_gradient(weights_tensor)

        difference = np.sum(np.abs(weights_tensor - np.ones(self.shape) * self.regularizer_strength))
        self.assertLessEqual(difference, 1e-10)

    def test_L2_norm(self):
        regularizer = Constraints.L2_Regularizer(self.regularizer_strength)

        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] += 1
        norm = regularizer.norm(weights_tensor)
        self.assertAlmostEqual(norm, np.sqrt(32) * self.regularizer_strength)

    def test_L1_with_sgd(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.Sgd(2)
        regularizer = Constraints.L1_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), -116, 2)

    def test_L2_with_sgd(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.Sgd(2)
        regularizer = Constraints.L2_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), 268, 2)

    def test_L1_with_sgd_w_momentum(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.SgdWithMomentum(2, .9)
        regularizer = Constraints.L1_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), -188, 1)

    def test_L2_with_sgd_w_momentum(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.SgdWithMomentum(2, .9)
        regularizer = Constraints.L2_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), 196, 1)

    def test_L1_with_adam(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.Adam(2, .9, .999)
        regularizer = Constraints.L1_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), -68, 2)

    def test_L2_with_adam(self):
        weights_tensor = np.ones(self.shape)
        weights_tensor[1:3, 2:4] *= -1

        optimizer = Optimizers.Adam(2, .9, .999)
        regularizer = Constraints.L2_Regularizer(2)
        optimizer.add_regularizer(regularizer)

        result = optimizer.calculate_update(weights_tensor, np.ones(self.shape)*2)
        result = optimizer.calculate_update(result, np.ones(self.shape) * 2)

        np.testing.assert_almost_equal(np.sum(result), 188, 2)


class TestDropout(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10000
        self.input_size = 10

        self.input_tensor = np.ones((self.batch_size, self.input_size))

    def test_forward_trainTime(self):
        drop_layer = Dropout.Dropout(0.25)
        output = drop_layer.forward(self.input_tensor)
        self.assertEqual(np.max(output), 4)
        self.assertEqual(np.min(output), 0)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertAlmostEqual(sum_over_mean/self.input_size, 1. , places=1)


    def test_position_preservation(self):
        drop_layer = Dropout.Dropout(0.5)
        output = drop_layer.forward(self.input_tensor)
        error_prev = drop_layer.backward(self.input_tensor)
        np.testing.assert_almost_equal(np.where(output==0.),np.where(error_prev==0.))

    def test_forward_testTime(self):
        drop_layer = Dropout.Dropout(0.5)
        drop_layer.phase = Base.Phase.test
        output = drop_layer.forward(self.input_tensor)

        self.assertEqual(np.max(output), 1.)
        self.assertEqual(np.min(output), 1.)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertEqual(sum_over_mean, 1. * self.input_size)

    def test_backward(self):
        drop_layer = Dropout.Dropout(0.5)
        drop_layer.forward(self.input_tensor)
        output = drop_layer.backward(self.input_tensor)

        self.assertEqual(np.max(output), 1)
        self.assertEqual(np.min(output), 0)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertAlmostEqual(sum_over_mean, .5 * self.input_size, places=1)


class TestBatchNorm(unittest.TestCase):
    plot = False
    directory = 'plots/'

    def setUp(self):
        self.batch_size = 200
        self.channels = 2
        self.input_shape = (self.channels, 3, 3)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(0)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        self.input_tensor_conv = np.abs(np.random.random((self.batch_size, *self.input_shape)))

        self.categories = 5
        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(Flatten.Flatten())
        self.layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        self.layers.append(L2Loss())

        self.plot_shape = (self.input_shape[1], self.input_shape[0] * np.prod(self.input_shape[2:]))

    @staticmethod
    def _channel_moments(tensor, channels):

        tensor = np.transpose(tensor, (0, *range(2, tensor.ndim), 1))
        tensor = tensor.reshape(-1, channels)
        mean = np.mean(tensor, axis=0)
        var = np.var(tensor, axis=0)
        return mean, var

    def test_forward_shape(self):
        layer = BatchNormalization.BatchNormalization(self.input_tensor.shape[-1])
        output = layer.forward(self.input_tensor)

        self.assertEqual(output.shape[0], self.input_tensor.shape[0])
        self.assertEqual(output.shape[1], self.input_tensor.shape[1])

    def test_forward_shape_convolutional(self):
        layer = BatchNormalization.BatchNormalization(self.channels)
        output = layer.forward(self.input_tensor_conv)

        self.assertEqual(output.shape, self.input_tensor_conv.shape)

    def test_forward(self):
        layer = BatchNormalization.BatchNormalization(self.input_tensor.shape[-1])
        output = layer.forward(self.input_tensor)
        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        self.assertAlmostEqual(np.sum(np.square(mean - np.zeros(mean.shape[0]))), 0)
        self.assertAlmostEqual(np.sum(np.square(var - np.ones(var.shape[0]))), 0)

    def test_reformat_image2vec(self):
        layer = BatchNormalization.BatchNormalization(3)
        layer.input_shape = (5, 3, 6, 4)
        image_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4)
        vec_tensor = layer.reformat(image_tensor)
        np.testing.assert_equal(vec_tensor.shape, (120, 3))
        self.assertEqual(np.sum(vec_tensor, 1)[0], 72)
        self.assertEqual(np.sum(vec_tensor, 0)[0], 18660)

    def test_reformat_vec2image(self):
        layer = BatchNormalization.BatchNormalization(3)
        layer.input_shape = (5, 3, 6, 4)
        vec_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(120, 3)
        image_tensor = layer.reformat(vec_tensor)
        np.testing.assert_equal(image_tensor.shape, (5, 3, 6, 4))
        self.assertEqual(np.sum(image_tensor, (0,1,2))[0], 15750)
        self.assertEqual(np.sum(image_tensor, (0,2,3))[0], 21420)

    def test_reformat(self):
        layer = BatchNormalization.BatchNormalization(3)
        layer.input_shape = (5, 3, 6, 4)
        image_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4)
        vec_tensor = layer.reformat(image_tensor)
        image_tensor2 = layer.reformat(vec_tensor)
        np.testing.assert_allclose(image_tensor, image_tensor2)


    def test_forward_convolutional(self):
        layer = BatchNormalization.BatchNormalization(self.channels)
        output = layer.forward(self.input_tensor_conv)
        mean, var = TestBatchNorm._channel_moments(output, self.channels)

        self.assertAlmostEqual(np.sum(np.square(mean)), 0)
        self.assertAlmostEqual(np.sum(np.square(var - np.ones_like(var))), 0)

    def test_forward_train_phase(self):
        layer = BatchNormalization.BatchNormalization(self.input_tensor.shape[-1])
        layer.forward(self.input_tensor)

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertNotEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)

    def test_forward_train_phase_convolutional(self):
        layer = BatchNormalization.BatchNormalization(self.channels)
        layer.forward(self.input_tensor_conv)

        output = layer.forward((np.zeros_like(self.input_tensor_conv)))

        mean, var = TestBatchNorm._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm._channel_moments(self.input_tensor_conv, self.channels)

        self.assertNotEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)

    def test_forward_test_phase(self):
        layer = BatchNormalization.BatchNormalization(self.input_tensor.shape[-1])
        layer.forward(self.input_tensor)
        layer.phase = Base.Phase.test

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertAlmostEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)
        self.assertAlmostEqual(np.sum(np.square(var)), 0)

    def test_forward_test_phase_convolutional(self):
        layer = BatchNormalization.BatchNormalization(self.channels)
        layer.forward(self.input_tensor_conv)
        layer.phase = Base.Phase.test

        output = layer.forward((np.zeros_like(self.input_tensor_conv)))

        mean, var = TestBatchNorm._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm._channel_moments(self.input_tensor_conv, self.channels)

        self.assertAlmostEqual(np.sum(np.square(mean + (mean_input / np.sqrt(var_input)))), 0)
        self.assertAlmostEqual(np.sum(np.square(var)), 0)

    def test_gradient(self):
        self.layers[0] = BatchNormalization.BatchNormalization(self.input_tensor.shape[-1])
        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_gradient_weights(self):
        self.layers[0] = BatchNormalization.BatchNormalization(self.input_tensor.shape[-1])
        self.layers[0].forward(self.input_tensor)
        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_bias(self):
        self.layers[0] = BatchNormalization.BatchNormalization(self.input_tensor.shape[-1])
        self.layers[0].forward(self.input_tensor)
        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, True)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_convolutional(self):
        self.layers[0] = BatchNormalization.BatchNormalization(self.channels)
        difference = Helpers.gradient_check(self.layers, self.input_tensor_conv, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-3)

    def test_gradient_weights_convolutional(self):
        self.layers[0] = BatchNormalization.BatchNormalization(self.channels)
        self.layers[0].forward(self.input_tensor_conv)
        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor_conv, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_bias_convolutional(self):
        self.layers[0] = BatchNormalization.BatchNormalization(self.channels)
        self.layers[0].forward(self.input_tensor_conv)
        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor_conv, self.label_tensor, True)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_update(self):
        layer = BatchNormalization.BatchNormalization(self.input_tensor.shape[-1])
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros_like(self.input_tensor)
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))


if RNN_TEST:
    class TestRNN(unittest.TestCase):
        def setUp(self):
            self.batch_size = 9
            self.input_size = 13
            self.output_size = 5
            self.hidden_size = 7
            self.input_tensor = np.random.rand(self.input_size, self.batch_size).T

            self.categories = 4
            self.label_tensor = np.zeros([self.categories, self.batch_size]).T
            for i in range(self.batch_size):
                self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        def test_forward_size(self):
            layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
            output_tensor = layer.forward(self.input_tensor)
            self.assertEqual(output_tensor.shape[1], self.output_size)
            self.assertEqual(output_tensor.shape[0], self.batch_size)

        def test_forward_stateful(self):
            layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)

            input_vector = np.random.rand(self.input_size, 1).T
            input_tensor = np.tile(input_vector, (2, 1))

            output_tensor = layer.forward(input_tensor)

            self.assertNotEqual(np.sum(np.square(output_tensor[0, :] - output_tensor[1, :])), 0)

        def test_forward_stateful_TBPTT(self):
            layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
            layer.memorize = True
            output_tensor_first = layer.forward(self.input_tensor)
            output_tensor_second = layer.forward(self.input_tensor)
            self.assertNotEqual(np.sum(np.square(output_tensor_first - output_tensor_second)), 0)

        def test_backward_size(self):
            layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = layer.backward(output_tensor)
            self.assertEqual(error_tensor.shape[1], self.input_size)
            self.assertEqual(error_tensor.shape[0], self.batch_size)

        def test_update(self):
            layer = RNN.RNN(self.input_size, self.hidden_size, self.output_size)
            layer.optimizer = Optimizers.Sgd(1)
            for _ in range(10):
                output_tensor = layer.forward(self.input_tensor)
                error_tensor = np.zeros([self.output_size, self.batch_size]).T
                error_tensor -= output_tensor
                layer.backward(error_tensor)
                new_output_tensor = layer.forward(self.input_tensor)
                self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

        def test_gradient(self):
            input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
            layers = list()
            layer = RNN.RNN(self.input_size, self.hidden_size, self.categories)
            layer.initialize(Initializers.He(), Initializers.He())
            layers.append(layer)
            layers.append(L2Loss())
            difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
            self.assertLessEqual(np.sum(difference), 1e-4)

        def test_gradient_weights(self):
            input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
            layers = list()
            layer = RNN.RNN(self.input_size, self.hidden_size, self.categories)
            layer.initialize(Initializers.He(), Initializers.He())
            layers.append(layer)
            layers.append(L2Loss())
            difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
            self.assertLessEqual(np.sum(difference), 1e-4)

        def test_weights_shape(self):
            layer = RNN.RNN(self.input_size, self.hidden_size, self.categories)
            layer.initialize(Initializers.He(), Initializers.He())
            self.assertTrue(hasattr(layer, 'weights'), msg='your RNN layer does not provide a weights attribute')
            fc_layer = FullyConnected.FullyConnected(20, 7)
            self.assertIn(layer.weights.shape, [fc_layer.weights.shape, fc_layer.weights.T.shape])

        def test_bias(self):
            input_tensor = np.zeros((1, 100000))
            layer = RNN.RNN(100000, 100, 1)
            layer.initialize(Initializers.UniformRandom(), Initializers.UniformRandom())
            result = layer.forward(input_tensor)
            self.assertGreater(np.sum(result), 0)


class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 13
        self.output_size = 5
        self.hidden_size = 7
        self.input_tensor = np.random.rand(self.input_size, self.batch_size).T

        self.categories = 4
        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_size(self):
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_forward_stateful(self):
        bptt_length = 2
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.output_size)

        input_vector = np.random.rand(1, self.input_size)
        input_tensor = np.tile(input_vector, (bptt_length, 1))

        output_tensor = layer.forward(input_tensor)

        self.assertNotEqual(np.sum(np.square(output_tensor[0, :] - output_tensor[1, :])), 0)

    def test_weights_shape(self):
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.categories)
        layer.initialize(Initializers.He(), Initializers.He())
        self.assertTrue(hasattr(layer, 'weights'), msg='your LSTM layer does not provide a weights member')
        fc_layer = FullyConnected.FullyConnected(20, 28)
        self.assertIn(layer.weights.shape, [fc_layer.weights.shape, fc_layer.weights.T.shape])

    def test_forward_stateful_TBPTT(self):
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.output_size)
        layer.memorize = True
        output_tensor_first = layer.forward(self.input_tensor)
        output_tensor_second = layer.forward(self.input_tensor)
        self.assertNotEqual(np.sum(np.square(output_tensor_first - output_tensor_second)), 0)

    def test_backward_size(self):
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([self.output_size, self.batch_size]).T
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        layers = list()
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.categories)
        layer.initialize(Initializers.He(), Initializers.He())
        layers.append(layer)
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        layers = list()
        layer = LSTM.LSTM(self.input_size, self.hidden_size, self.categories)
        layer.initialize(Initializers.He(), Initializers.He())
        layers.append(layer)
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-3)

    def test_bias(self):
        bptt_length = 1
        input_tensor = np.zeros((1, 100000))
        layer = LSTM.LSTM(100000, 100, 1)
        layer.initialize(Initializers.UniformRandom(), Initializers.UniformRandom())
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)


class TestNeuralNetwork(unittest.TestCase):
    plot = False
    directory = 'plots/'
    log = 'log.txt'

    def test_data_access(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-4),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.layers.append(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.layers.append(fcl_2)
        net.layers.append(SoftMax.SoftMax())

        out = net.forward()
        out2 = net.forward()

        self.assertNotEqual(out, out2)

    def test_iris_data(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-3),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_trainable_layer(fcl_2)
        net.layers.append(SoftMax.SoftMax())

        net.train(4000)
        if TestNeuralNetwork.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_momentum(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.SgdWithMomentum(1e-3, 0.8),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_trainable_layer(fcl_2)
        net.layers.append(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Momentum')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork_Momentum.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_adam(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-3, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_trainable_layer(fcl_2)
        net.layers.append(SoftMax.SoftMax())

        net.train(3000)
        if TestNeuralNetwork.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using ADAM')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork_ADAM.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_batchnorm(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        net.append_trainable_layer(BatchNormalization.BatchNormalization(input_size))
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_trainable_layer(fcl_2)
        net.layers.append(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Batchnorm')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork_Batchnorm.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        results_next_run = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset using Batchnorm, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertGreater(accuracy, 0.8)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)

    def test_iris_data_with_dropout(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_trainable_layer(fcl_1)
        net.layers.append(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_trainable_layer(fcl_2)
        net.layers.append(Dropout.Dropout(0.3))
        net.layers.append(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Dropout')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork_Dropout.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)

        results_next_run = net.test(data)

        with open(self.log, 'a') as f:
            print('On the Iris dataset using Dropout, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)


class TestConvNet(unittest.TestCase):
    plot = False
    directory = 'plots/'
    log = 'log.txt'
    iterations = 100

    def test_digit_data(self):
        adam = Optimizers.Adam(5e-3, 0.98, 0.999)
        self._perform_test(adam, TestConvNet.iterations, 'ADAM', False, False)

    def test_digit_data_L2_Regularizer(self):
        sgd_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(sgd_with_l2, TestConvNet.iterations, 'L2_regularizer', False, False)

    def test_digit_data_L1_Regularizer(self):
        sgd_with_l1 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l1.add_regularizer(Constraints.L1_Regularizer(8e-2))
        self._perform_test(sgd_with_l1, TestConvNet.iterations, 'L1_regularizer', False, False)

    def test_digit_data_dropout(self):
        sgd_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(4e-4))
        self._perform_test(sgd_with_l2, TestConvNet.iterations, 'Dropout', True, False)

    def test_digit_batch_norm(self):
        adam = Optimizers.Adam(1e-2, 0.98, 0.999)
        self._perform_test(adam, TestConvNet.iterations, 'Batch_norm', False, True)

    def test_all(self):
        sgd_with_l2 = Optimizers.Adam(1e-2, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(sgd_with_l2, TestConvNet.iterations, 'Batch_norm and L2', False, True)

    def _perform_test(self, optimizer, iterations, description, dropout, batch_norm):
        net = NeuralNetwork.NeuralNetwork(optimizer,
                                          Initializers.He(),
                                          Initializers.Constant(0.1))
        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 150
        num_kernels = 4

        net.data_layer = Helpers.DigitData(batch_size)
        net.loss_layer = Loss.CrossEntropyLoss()

        if batch_norm:
            net.append_trainable_layer(BatchNormalization.BatchNormalization(1))

        cl_1 = Conv.Conv(conv_stride_shape, convolution_shape, num_kernels)
        net.append_trainable_layer(cl_1)
        cl_1_output_shape = (num_kernels, *input_image_shape[1:])

        if batch_norm:
            net.append_trainable_layer(BatchNormalization.BatchNormalization(num_kernels))

        net.layers.append(ReLU.ReLU())

        fcl_1_input_size = np.prod(cl_1_output_shape)

        net.layers.append(Flatten.Flatten())

        fcl_1 = FullyConnected.FullyConnected(fcl_1_input_size, np.int(fcl_1_input_size/2.))
        net.append_trainable_layer(fcl_1)

        if batch_norm:
            net.append_trainable_layer(BatchNormalization.BatchNormalization(fcl_1_input_size//2))

        if dropout:
            net.layers.append(Dropout.Dropout(0.3))

        net.layers.append(ReLU.ReLU())

        fcl_2 = FullyConnected.FullyConnected(np.int(fcl_1_input_size / 2), np.int(fcl_1_input_size / 3))
        net.append_trainable_layer(fcl_2)

        net.layers.append(ReLU.ReLU())

        fcl_3 = FullyConnected.FullyConnected(np.int(fcl_1_input_size / 3), categories)
        net.append_trainable_layer(fcl_3)

        net.layers.append(SoftMax.SoftMax())

        net.train(iterations)
        if TestConvNet.plot:
            fig = plt.figure('Loss function for training a Convnet on the Digit dataset ' + description)
            plt.plot(net.loss, '-x')
            plt.ylim(ymin=0, ymax=450)
            fig.savefig(os.path.join(self.directory, "TestConvNet_" + description + ".pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(description, accuracy * 100.), file=f)
        print('\nOn the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(description, accuracy * 100.))
        self.assertGreater(accuracy, 0.3)


class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)


if __name__ == "__main__":
    plot = True
    directory = 'plots'
    log = 'log.txt'
    with open(log, 'a') as f:
        print('Run ' + str(datetime.datetime.now()), file=f)
    if not os.path.exists(directory):
        os.makedirs(directory)
    TestNeuralNetwork.plot = plot
    TestConv.plot = plot
    TestPooling.plot = plot
    TestConvNet.plot = plot
    TestBatchNorm.plot = plot
    TestNeuralNetwork.directory = directory
    TestConv.directory = directory
    TestPooling.directory = directory
    TestConvNet.directory = directory
    TestBatchNorm.directory = directory
    TestNeuralNetwork.log = log
    TestConv.log = log
    unittest.main()
    with open(log, 'a') as f:
        print('Finished Tests' + str(datetime.datetime.now()), file=f)
