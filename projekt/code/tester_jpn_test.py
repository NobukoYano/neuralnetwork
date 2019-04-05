# -*- coding: utf-8 -*-
"""
This module is for Testing of handwritten Japanese character.
Test for 3 layer neural network with
784 input, 200 hidden and 5 output nodes
(5 Outputs = Hiragana of a, e, i, o, u)

Author : Nobuko Yano
Version : 1.0
Last Updated : 2018-11-20

Attributes:
    x(int)       : number of input nodes
    h(int)       : number of hidden nodes
    y(int)       : number of output nodes
    lr(float)    : learning rate
    file_train(str)  : filename of test data for train
    file_test(str)  : filename of test data for train

Configuration
    'config.ini'
    [GENERAL]
    [JPN_DIR] : directory of test data

"""

import configparser
import glob
import os
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuralnetwork2 import NeuralNetwork

if __name__ == '__main__':

    x = 784  # number of input nodes
    h = 200  # number of hidden nodes
    y = 5  # number of output nodes
    lr = 0.1  # learning rate

    # Read directory form config.ini
    inifile = configparser.ConfigParser()
    inifile.read('config.ini')
    PATH = os.path.join(Path(__file__).resolve().parents[1],
                        inifile['GENERAL']['JPN_DIR'])
    file_train = 'jpn_train.csv'
    file_test = 'jpn_test.csv'

    # initialise NeuralNetwork
    nn = NeuralNetwork(x, h, y, lr)

    # loading training data
    load_train = []
    for image_file_name in glob.glob('../test/jpn/0??_?.png'):
        # use the filename to set the correct label
        label = int(image_file_name[-5:-4])
        # load image data from png files into an array
        img_array = imageio.imread(image_file_name, as_gray=True)
        # reshape from 28x28 to list of 784 values, invert values
        img_data = 255.0 - img_array.reshape(784)
        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01
        # append label and image data  to test data set
        record = np.append(label, img_data)
        # print(record)
        load_train.append(record)
        pass

    pd.DataFrame(data=load_train).to_csv(
        os.path.join(PATH, file_train), sep=',', index=False)

    # loading testing data
    load_test = []
    for image_file_name in glob.glob('../test/jpn/1??_?.png'):
        # use the filename to set the correct label
        label = int(image_file_name[-5:-4])
        # load image data from png files into an array
        img_array = imageio.imread(image_file_name, as_gray=True)
        # reshape from 28x28 to list of 784 values, invert values
        img_data = 255.0 - img_array.reshape(784)
        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01
        # append label and image data  to test data set
        record = np.append(label, img_data)
        # print(record)
        load_test.append(record)
        pass

    pd.DataFrame(data=load_test).to_csv(
        os.path.join(PATH, file_test), sep=',', index=False)

    # epochs is the number of times the training data set is used for training
    epochs = 10

    performance_array = []
    for epoch in range(epochs):
        for e in range(epoch):
            # go through all records in the training data set
            for record in load_train:
                # split the record by the ',' commas
                all_values = record
                # all_values = record.split(',')
                # scale and shift the inputs
                inputs = np.asfarray(all_values[1:])
                # create the target output values
                targets = np.zeros(y) + 0.01
                # all_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99
                nn.train(inputs, targets)
                pass
            pass

        # scorecard for how well the network performs, initially empty
        scorecard = []

        # go through all the records in the test data set
        for record in load_test:
            # split the record by the ',' commas
            all_values = record
            # correct answer is first value
            correct_label = int(all_values[0])
            # scale and shift the inputs
            inputs = np.asfarray(all_values[1:])
            # query the network
            outputs = nn.query(inputs)
            # the index of the highest value corresponds to the label
            label = np.argmax(outputs)
            # append correct or incorrect to list
            if (label == correct_label):
                # network's answer matches correct answer, add 1 to scorecard
                scorecard.append(1)
            else:
                # network's answer doesn't match correct answer, add 0
                scorecard.append(0)

        # calculate the performance score, the fraction of correct answers
        scorecard_array = np.asarray(scorecard)
        performance = scorecard_array.sum() / scorecard_array.size
        performance_array.append(performance)
        print("performance = ", performance)

    # Show performance in Chart
    plt.title('Performance von Erkennung der Japanischen Zeichen')
    plt.xlabel('Anzahl von Epoche')
    plt.ylabel('Treffquote')
    plt.plot(performance_array, label="Durchschnittliche Treffquote")
    plt.legend()
    plt.show()
    print(performance_array)
