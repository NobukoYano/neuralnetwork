# -*- coding: utf-8 -*-
"""
This module include class NeuralNetwork of 2 layers
Weights must be optimised one of the following options
    - set from standard distribution, mean 0.0
    - set from random number from 0.01 to 0.99
    - set fixed value

Author : Nobuko Yano
Version : 1.0
Last Updated : 2018-11-21

Attributes:
    inodes(int)    : number of input nodes
    onodes(int)    : number of output nodes
    lr(float)      : learning rate
    errors(ndarray): difference between target and predicted value
    weights(ndarray) : weights array of size (self.onodes * self.inodes)
    activation_function : calculate outputs with sigmoid function

Configuration
    n/a

"""

import numpy as np
import scipy.special


class NeuralNetwork():

    # Initialise the neural network
    def __init__(self, inputnodes, outputnodes, learningrate):
        # Set number of nodes
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.lr = learningrate

        # Initialise error
        self.errors = None

        # For Test 01-002, 01-005, 01-007
        # Set random number with normal distribution with mean = 0
        # self.weight = np.random.normal(0.0, pow(self.inodes, -0.5),
        #                               (self.onodes, self.inodes))

        # For Test 01-003, 01-006, 01-008
        # Set random number between 0.01 - 0.99
        self.weight = np.random.randint(1, 99, (self.onodes,
                                                self.inodes)) / 100

        # Set with hard coding
        # self.weight = np.array([[0.9, 0.3], [0.2, 0.8]])

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into final output layer
        final_inputs = np.dot(self.weight, inputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        self.output_errors = targets - final_outputs
        # self.errors = np.append(self.errors, output_errors)

        # update the weights
        self.weight += self.lr * np.dot((self.output_errors *
                                         final_outputs *
                                         (1.0 - final_outputs)),
                                        np.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        pass
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into final output layer
        final_inputs = np.dot(self.weight, inputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
