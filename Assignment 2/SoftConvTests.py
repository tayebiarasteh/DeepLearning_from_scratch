from Layers import *

import numpy as np

# determines whether difference results are printed to stdout
print_differences = True

# 4D array with dimension [1, 1, 14, 14]
base_input_image = np.array([[
    [[1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ],
     [1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1. ],
     [1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. ],
     [1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. ],
     [1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. ],
     [1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1. ],
     [1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1. ],
     [1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1. ],
     [1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1. ],
     [1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. ],
     [1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. ],
     [1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. ],
     [1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1. ],
     [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ]]]])

# define two sets of kernels
weights_c1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 2.0
weights_c2 = np.copy(weights_c1.T)

# make sure weights have "channel" dimension
weights_c1 = weights_c1[np.newaxis]
weights_c2 = weights_c2[np.newaxis]

# combine weights to one tensor
weights = np.array([weights_c1, weights_c2])

expected_result = np.array([[
    [[ 1.5 ,  0.  ,  0.  ,  0.  , -0.25, -0.25,  0.,  0.,  0.25,  0.25,  0.  ,  0.  ,  0.  , -1.5 ],
     [ 2.  ,  0.  , -0.25, -0.25, -0.5 , -0.5 ,  0.,  0.,  0.5 ,  0.5 ,  0.25,  0.25,  0.  , -2.  ],
     [ 2.  , -0.25, -0.75, -0.5 , -0.25, -0.25,  0.,  0.,  0.25,  0.25,  0.5 ,  0.75,  0.25, -2.  ],
     [ 2.  , -0.75, -1.  , -0.25,  0.  ,  0.  ,  0.,  0.,  0.  ,  0.  ,  0.25,  1.  ,  0.75, -2.  ],
     [ 1.75, -1.  , -0.75,  0.  , -0.25, -0.25,  0.,  0.,  0.25,  0.25,  0.  ,  0.75,  1.  , -1.75],
     [ 1.25, -1.  , -0.25,  0.  , -0.75, -0.75,  0.,  0.,  0.75,  0.75,  0.  ,  0.25,  1.  , -1.25],
     [ 1.  , -1.  ,  0.  ,  0.  , -1.  , -1.  ,  0.,  0.,  1.  ,  1.  ,  0.  ,  0.  ,  1.  , -1.  ],
     [ 1.  , -1.  ,  0.  ,  0.  , -1.  , -1.  ,  0.,  0.,  1.  ,  1.  ,  0.  ,  0.  ,  1.  , -1.  ],
     [ 1.25, -1.  , -0.25,  0.  , -0.75, -0.75,  0.,  0.,  0.75,  0.75,  0.  ,  0.25,  1.  , -1.25],
     [ 1.75, -1.  , -0.75,  0.  , -0.25, -0.25,  0.,  0.,  0.25,  0.25,  0.  ,  0.75,  1.  , -1.75],
     [ 2.  , -0.75, -1.  , -0.25,  0.  ,  0.  ,  0.,  0.,  0.  ,  0.  ,  0.25,  1.  ,  0.75, -2.  ],
     [ 2.  , -0.25, -0.75, -0.5 , -0.25, -0.25,  0.,  0.,  0.25,  0.25,  0.5 ,  0.75,  0.25, -2.  ],
     [ 2.  ,  0.  , -0.25, -0.25, -0.5 , -0.5 ,  0.,  0.,  0.5 ,  0.5 ,  0.25,  0.25,  0.  , -2.  ],
     [ 1.5 ,  0.  ,  0.  ,  0.  , -0.25, -0.25,  0.,  0.,  0.25,  0.25,  0.  ,  0.  ,  0.  , -1.5 ]],
    [[ 1.5 ,  2.  ,  2.  ,  2.  ,  1.75,  1.25,  1.,  1.,  1.25,  1.75,  2.  ,  2.  ,  2.  ,  1.5 ],
     [ 0.  ,  0.  , -0.25, -0.75, -1.  , -1.  , -1., -1., -1.  , -1.  , -0.75, -0.25,  0.  ,  0.  ],
     [ 0.  , -0.25, -0.75, -1.  , -0.75, -0.25,  0.,  0., -0.25, -0.75, -1.  , -0.75, -0.25,  0.  ],
     [ 0.  , -0.25, -0.5 , -0.25,  0.  ,  0.  ,  0.,  0.,  0.  ,  0.  , -0.25, -0.5 , -0.25,  0.  ],
     [-0.25, -0.5 , -0.25,  0.  , -0.25, -0.75, -1., -1., -0.75, -0.25,  0.  , -0.25, -0.5 , -0.25],
     [-0.25, -0.5 , -0.25,  0.  , -0.25, -0.75, -1., -1., -0.75, -0.25,  0.  , -0.25, -0.5 , -0.25],
     [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.,  0.,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
     [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.,  0.,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
     [ 0.25,  0.5 ,  0.25,  0.  ,  0.25,  0.75,  1.,  1.,  0.75,  0.25,  0.  ,  0.25,  0.5 ,  0.25],
     [ 0.25,  0.5 ,  0.25,  0.  ,  0.25,  0.75,  1.,  1.,  0.75,  0.25,  0.  ,  0.25,  0.5 ,  0.25],
     [ 0.  ,  0.25,  0.5 ,  0.25,  0.  ,  0.  ,  0.,  0.,  0.  ,  0.  ,  0.25,  0.5 ,  0.25,  0.  ],
     [ 0.  ,  0.25,  0.75,  1.  ,  0.75,  0.25,  0.,  0.,  0.25,  0.75,  1.  ,  0.75,  0.25,  0.  ],
     [ 0.  ,  0.  ,  0.25,  0.75,  1.  ,  1.  ,  1.,  1.,  1.  ,  1.  ,  0.75,  0.25,  0.  ,  0.  ],
     [-1.5 , -2.  , -2.  , -2.  , -1.75, -1.25, -1., -1., -1.25, -1.75, -2.  , -2.  , -2.  , -1.5 ]]]])

# dummy error tensor
err_next = np.repeat(base_input_image, 2, axis=1) / 4
err_next[0, 1] *= -1

expected_err_prev = np.array([[
    [[ 0.   ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.375,  0.25,  0.25,  0.25 ,  0.375,  0.5  ,  0.5  ,  0.5  ,  0.75 ],
     [-0.5  ,  0.   ,  0.   , -0.125, -0.125, -0.125, -0.25, -0.25, -0.375, -0.375, -0.25 , -0.125,  0.   ,  0.5  ],
     [-0.5  ,  0.   ,  0.   , -0.125, -0.125,  0.   ,  0.  ,  0.  , -0.125, -0.25 , -0.375, -0.375, -0.125,  0.5  ],
     [-0.5  ,  0.125,  0.125,  0.   ,  0.   ,  0.   ,  0.  ,  0.  ,  0.   ,  0.   , -0.125, -0.375, -0.25 ,  0.5  ],
     [-0.5  ,  0.125,  0.125,  0.   ,  0.   , -0.125, -0.25, -0.25, -0.25 , -0.125,  0.   , -0.25 , -0.375,  0.375],
     [-0.375,  0.125,  0.   ,  0.   ,  0.125,  0.   , -0.25, -0.25, -0.375, -0.25 ,  0.   , -0.125, -0.375,  0.25 ],
     [-0.25 ,  0.25 ,  0.   ,  0.   ,  0.25 ,  0.25 ,  0.  ,  0.  , -0.25 , -0.25 ,  0.   ,  0.   , -0.25 ,  0.25 ],
     [-0.25 ,  0.25 ,  0.   ,  0.   ,  0.25 ,  0.25 ,  0.  ,  0.  , -0.25 , -0.25 ,  0.   ,  0.   , -0.25 ,  0.25 ],
     [-0.25 ,  0.375,  0.125,  0.   ,  0.25 ,  0.375,  0.25,  0.25,  0.   , -0.125,  0.   ,  0.   , -0.125,  0.375],
     [-0.375,  0.375,  0.25 ,  0.   ,  0.125,  0.25 ,  0.25,  0.25,  0.125,  0.   ,  0.   , -0.125, -0.125,  0.5  ],
     [-0.5  ,  0.25 ,  0.375,  0.125,  0.   ,  0.   ,  0.  ,  0.  ,  0.   ,  0.   ,  0.   , -0.125, -0.125,  0.5  ],
     [-0.5  ,  0.125,  0.375,  0.375,  0.25 ,  0.125,  0.  ,  0.  ,  0.   ,  0.125,  0.125,  0.   ,  0.   ,  0.5  ],
     [-0.5  ,  0.   ,  0.125,  0.25 ,  0.375,  0.375,  0.25,  0.25,  0.125,  0.125,  0.125,  0.   ,  0.   ,  0.5  ],
     [-0.75 , -0.5  , -0.5  , -0.5  , -0.375, -0.25 , -0.25, -0.25, -0.375, -0.5  , -0.5  , -0.5  , -0.5  ,  0.   ]]]])

expected_gradient_bias = np.array([33., -33.])
expected_gradient_weights = np.array([[
    [[ 18.75,  22.5 ,  18.75],
     [ 22.5 ,  27.  ,  22.5 ],
     [ 18.75,  22.5 ,  18.75]]],
   [[[-18.75, -22.5 , -18.75],
     [-22.5 , -27.  , -22.5 ],
     [-18.75, -22.5 , -18.75]]]])


def test_conv_forward_2d(bias, times=1):

    conv_layer = Conv.Conv((1, 1), (1, 3, 3), 2)
    conv_layer.weights = weights
    conv_layer.bias = bias

    output_forward = None
    for t in range(times):
        output_forward = conv_layer.forward(base_input_image)

    expected_result_bias = np.zeros_like(expected_result)
    expected_result_bias[:, 0] = expected_result[:, 0] + bias[0]
    expected_result_bias[:, 1] = expected_result[:, 1] + bias[1]

    print("Expected shape: {}".format(expected_result_bias.shape))
    print("Actual   shape: {}".format(output_forward.shape), flush=True)
    # make sure message is printed before the assert

    assert expected_result.shape == output_forward.shape, "The shape of result of the forward pass is not correct."

    if print_differences:
        print("Difference between expected and real result:\n{}".format(expected_result_bias - output_forward), flush=True)

    assert np.sum(np.abs(expected_result_bias - output_forward)) < 1e-7, "Result of the forward pass is not correct."


def test_conv_backward_2d(times=1):
    conv_layer = Conv.Conv((1, 1), (1, 3, 3), 2)
    conv_layer.weights = weights
    conv_layer.bias = np.array([0.5, 1])

    err_prev = None
    for i in range(times):
        # forward pass through the Conv-layer
        output_forward = conv_layer.forward(np.copy(base_input_image))

        # backward pass through the Conv-layer
        err_prev = conv_layer.backward(np.copy(err_next))

    print("Expected shape E_(n-1): {}".format(expected_err_prev.shape))
    print("Actual   shape E_(n-1): {}".format(err_prev.shape), flush=True)

    # Assert that the shapes match.
    assert expected_err_prev.shape == err_prev.shape, "Shape of the gradient with respect to the lower layers is not correct."

    if print_differences:
        print("Difference between expected and actual E_(n-1):\n{}".format(expected_err_prev - err_prev))
        print("Difference between expected and actual gradient weights:\n{}".format(expected_gradient_weights - conv_layer.gradient_weights))
        print("Difference between expected and actual gradient bias:\n{}".format(expected_gradient_bias - conv_layer.gradient_bias, flush=True))

    assert np.sum(np.abs(expected_gradient_bias - conv_layer.gradient_bias)) < 1e-7, \
        "Computation of error with respect to the bias is not correct."

    assert np.sum(np.abs(expected_err_prev - err_prev)) < 1e-7, "Computation of error with respect to the previous layer is not correct."

    assert np.sum(np.abs(expected_gradient_weights - conv_layer.gradient_weights)) < 1e-7, \
        "Computation of error with respect to the weight is not correct."


if __name__ == "__main__":
    """ READ THIS FIRST!
    "
    " The tests below are "soft" tests: They compare the results of your forward and backward pass with results computed by a correct implementation
    " and print this comparison to standard output.
    "
    " We specifically opted not to use the unittest framework here to make debugging easier.
    "
    " We recommend to comment out one test at a time. If the difference is above 0, debug your code step-by-step and fix the issue before continuing
    " to the next test.
    " Links: (if you use pycharm) https://www.jetbrains.com/help/pycharm/debugging-your-first-python-application.html
    "                             https://docs.python.org/2/library/pdb.html
    "
    " Note that these tests will not detect *all* implementation errors. They may serve as additional ideas on how to debug your code if these tests
    " all pass but the unit tests still do not.
    """

    # test if the forward pass works if the bias is set to zero
    test_conv_forward_2d(np.array([0, 0]))

    # test if the forward pass works if the bias is not equal to zero
    test_conv_forward_2d(np.array([0.5, -0.5]))

    # test if the forward pass works when it is called multiple times
    test_conv_forward_2d(np.array([0.5, -0.5]), 3)

    # test if the backward pass works when it is called multiple times
    test_conv_backward_2d()

    # test if the backward pass works when it is called multiple times
    test_conv_backward_2d(3)
