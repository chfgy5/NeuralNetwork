import numpy as np
import csv

def sigmoid(x):
    return 1 / (1 + np.e ** -x)

class neural_network:
    def __init__(self):
        self.inputs = np.array(self.get_given_values("./inputs/cross_data (3 inputs - 2 outputs).csv"))
        self.bias_hidden_layer = np.array(self.get_given_values("./inputs/b1 (11 nodes).csv"))
        self.weights_hidden_layer = np.array(self.get_given_values("./inputs/w1 (3 inputs - 11 nodes).csv"))
        self.bias_output_layer = np.array(self.get_given_values("./inputs/b2 (2 output nodes).csv"))
        self.weights_output_layer = np.array(self.get_given_values("./inputs/w2 (from 11 to 2).csv"))
        self.activation_function = sigmoid

    def get_given_values(self, filepath):
        array = []
        with open(filepath) as doc:
            csv_reader = csv.reader(doc, delimiter=',')
            for row in csv_reader:
                arr = []
                for item in row:
                    arr.append(float(item))
                array.append(arr)
        return array
    
    def train(self):
        for value in self.inputs:
            for weight in self.weights_hidden_layer:
                hidden_layer_induced_fields = np.dot(value[0:3], weight) + self.bias_hidden_layer
                #print(hidden_layer_induced_fields)
                hidden_layer_outputs = self.activation_function(hidden_layer_induced_fields)
                print(hidden_layer_outputs)
            break

nn = neural_network()
# print("inputs: ", nn.inputs)
# print("bias hidden: ", nn.bias_hidden_layer)
# print("weights hidden: ", nn.weights_hidden_layer)
# print("bias output: ", nn.bias_output_layer)
# print("weights output: ", nn.weights_output_layer)
# print("\n\n")
nn.train()
