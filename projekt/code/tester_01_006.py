# -*- coding: utf-8 -*-
"""
This module is for Testing No.01-006.
Test for 2 layer neural network with 2 input nodes and 2 output nodes

Author : Nobuko Yano
Version : 0.1
Last Updated : 2018-11-18

Attributes:
    x(int)       : number of input nodes
    y(int)       : number of output nodes
    lr(float)    : learning rate
    file_r(str)  : filename of test data for train
    file_qb(str) : filename of test data with predicted values before training
    file_qa(str) : filename of test data with predicted values after training
    file_w(str)  : filename of initial weights !!!NOT USED!!!
    i_from = 101 : Minimum value of inputs
    i_to = 1000  : Maximum value of inputs

Configuration
    'config.ini'
    [GENERAL]
    [TEST_DIR] : directory of test data

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuralnetwork import NeuralNetwork
from pathlib import Path
import os
import configparser

if __name__ == '__main__':
    # Set float format to 2 decimal places
    pd.options.display.float_format = '{:.4f}'.format

    x = 2  # number of input nodes
    #h = 2  # number of hidden nodes
    y = 2  # number of output nodes
    lr = 0.1  # learning rate
    i_from = 101
    i_to = 1000

    # Read directory form config.ini
    inifile = configparser.ConfigParser()
    inifile.read('config.ini')
    PATH = os.path.join(Path(__file__).resolve().parents[1], inifile['GENERAL']['TEST_DIR'])
    file_r = '20181108_01_b_0500-ergebnis.csv'
    file_qb = '20181108_01_b_0500_query_before_#6b.csv'
    file_qa = '20181108_01_b_0500_query_after_#6b.csv'
    #file_w = 'weight.csv'

    # initialise NeuralNetwork
    nn = NeuralNetwork(x, y, lr)

    # Set weights from csv file
    #nn.weight = pd.read_csv(os.path.join(PATH, file_w), sep=';', header=None)

    # Read csv file and split to input and output
    csv_all = pd.read_csv(os.path.join(PATH, file_r), sep=';')
    csv_input = (csv_all[['input1', 'input2']] - i_from) / (i_to - i_from)
    csv_output = csv_all[['output1', 'output2']]


    # Query data (predict data) from inputs
    csv_query = pd.DataFrame()
    for index, line in csv_input.iterrows():
        csv_query = csv_query.append(pd.DataFrame(data=np.array(nn.query(line), ndmin=2).T), ignore_index=True)
    #print('== query data before train ==')
    #print(csv_query)

    print('Initial Gewicht')
    print(nn.weight)
    #print(nn.weight_ih)
    #print(nn.weight_ho)

    # Add query data in data file (predict1 and predict2)
    result = pd.concat([csv_all, csv_query], axis=1)
    result.to_csv(os.path.join(PATH, file_qb), sep=';', index=False, float_format='%.4f',
                  header=['id', 'input1', 'input2', 'output1', 'output2', 'predict1', 'predict2'])


    # Train neural network and hold difference between outputs and predictions
    error = pd.DataFrame()
    for _ in range(1):
        for index, line in csv_all.iterrows():
            nn.train((line[['input1','input2']] - i_from) / (i_to - i_from),line[['output1','output2']])
            error = error.append(pd.DataFrame(data=np.array(nn.output_errors).T), ignore_index=True)

    #print(error)

    # Show errors in graph
    plt.title('Abweichung zwischen Zielwerte und simulierte Werte')
    plt.xlabel('Anzahl von Training')
    plt.ylabel('Abweichung')
    plt.plot(error[0], label="output1 - predict1")
    plt.plot(error[1], label="output2 - predict2")
    plt.legend()
    plt.show()

    # Query data (predict data) from inputs
    csv_query = pd.DataFrame()
    for index, line in csv_input.iterrows():
        csv_query = csv_query.append(pd.DataFrame(data=np.array(nn.query(line), ndmin=2).T), ignore_index=True)
    #print('== query data after train ==')
    #print(csv_query)

    # Add query data in data file (predict1 and predict2)
    result = pd.concat([csv_all, csv_query], axis=1)
    result.to_csv(os.path.join(PATH, file_qa), sep=';', index=False, float_format='%.4f',
                  header=['id', 'input1', 'input2', 'output1', 'output2', 'predict1', 'predict2'])

    print('Gewicht nach Training')
    print(nn.weight)
    #print(nn.weight_ih)
    #print(nn.weight_ho)
