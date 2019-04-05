# -*- coding: utf-8 -*-
"""
This module include class NeuralNetwork of 3 layers

Author : Nobuko Yano
Version : 1.0
Last Updated : 2018-11-21

Attributes:
    inodes(int): number of input nodes
    onodes(int): number of output nodes
    lr(float): learning rate
    errors(ndarray): difference between target and predicted value
    weights(ndarray) : weights  onodes * inodes
    activation_function : calculate outputs with sigmoid function

Configuration
    n/a

"""

import numpy as np
import scipy.special


class NeuralNetwork():

    # Initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Set number of nodes
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # Initialise error
        self.output_errors = None
        self.hidden_errors = None

        # Initialise weights
        # Set random number 0.0 to 1.0
        # self.weight_ih = np.random.rand(self.hnodes, self.inodes)
        # self.weight_ho = np.random.rand(self.onodes, self.hnodes)

        # For Test 01-102
        # Set random number with normal distribution with mean = 0
        self.weight_ih = np.random.normal(0.0, pow(self.inodes, -0.5),
                                          (self.hnodes, self.inodes))
        self.weight_ho = np.random.normal(0.0, pow(self.hnodes, -0.5),
                                          (self.onodes, self.hnodes))

        # For Test 01-101 01-103, 01-104, 01-105
        # Set random number between 0.01 - 0.99
        # self.weight_ih = np.random.randint(1, 99, (self.hnodes,
        #                                           self.inodes))/100
        # self.weight_ho = np.random.randint(1, 99, (self.onodes,
        #                                           self.hnodes))/100

        # Set with hard coding
        # self.weight = np.array([[0.9, 0.3], [0.2, 0.8]])

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.weight_ih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.weight_ho, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        self.output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights,
        # recombined at hidden nodes
        self.hidden_errors = np.dot(self.weight_ho.T, self.output_errors)

        # update the weights between output and hidden layers
        self.weight_ho += self.lr * np.dot((self.output_errors *
                                            final_outputs *
                                            (1.0 - final_outputs)),
                                           np.transpose(hidden_outputs))

        # update the weights between input and hidden layers
        self.weight_ih += self.lr * np.dot((self.hidden_errors *
                                            hidden_outputs *
                                            (1.0 - hidden_outputs)),
                                           np.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.weight_ih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.weight_ho, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
