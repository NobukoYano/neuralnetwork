from neuralnetwork import neuralNetwork


# number of input, hidden and output nodes
input_nodes = 3
hidden_nodes = 5
output_nodes = 3

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

print(n.who)
print(n.wih)

