# Deep Learning from scratch with NumPy

### By Soroosh Tayebi Arasteh

This project contains the programming exercises of the Deep Learning course (WS1920) offered by the [Pattern Recognition Lab (LME)](https://lme.tf.fau.de/) of the Computer Science department at University of Erlangen-Nuremberg (FAU).



## Contents
This repository contains 3 parts corresponding to the 3 exercises developed on top of each other for the course.

The main goal of the exercises was to develop fundamental blocks of neural networks such as different types of layers, optimizers and activations. This was done using good programming techniques such as efficient mathematical programming, object oriented programming, inheritance or polymorphism.

The project is written in **Python 3.7**. No additional deep learning framework is used and most of the layers and functions are implemented only using **NumPy**.

#### Overview of the project:


1. **Part 1 (Feed forward):** A `fully connected layer` object and `ReLU` & `Softmax` activations were developed as well as the `Cross Entropy` loss.

2. **Part 2 (Convolutional layer):** Basic blocks of Convolutional Neural Networks were devloped (`Conv` layer and `Pooling`). Several optimizers such as `SGD with Momentum` and `ADAM` were also developed.

3. **Part 3 (Recurrent layer and Regularization):** The classical `LSTM` Unit layer (basically the most used RNN architecture to date), which can be used in Recurrent Neural Networks, was developed (including more activations such as `TanH` or `Sigmoid`. Also different regularization layers (like `Batch Normalization`, `Dropout`, `L1` and `L2` regulizers) were developed.

The main class running everything is `NeuralNetwork.py`. Various unit tests for every layer and function are included in `NeuralNetworkTests.py`.

Further details such as task descriptions and specifications can be found inside ./Protocols directory.

