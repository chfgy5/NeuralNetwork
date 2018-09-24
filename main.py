import numpy as np
import csv

def sigmoid(x):
    return 1 / (1 + np.e ** -x)

class neural_network:
    def __init__(self):
        self.inputs = np.array(self.get_given_values("./inputs/cross_data (3 inputs - 2 outputs).csv"))
        self.bias_hidden_layer = np.array(self.get_given_values("./inputs/b1 (11 nodes).csv"))
        self.weights_hidden_layer = np.array(self.get_given_values("./inputs/w1 (3 inputs - 11 nodes).csv"))
        self.hidden_layer_outputs = np.zeros(11)
        self.bias_output_layer = np.array(self.get_given_values("./inputs/b2 (2 output nodes).csv"))
        self.weights_output_layer = np.array(self.get_given_values("./inputs/w2 (from 11 to 2).csv"))
        self.outputs = np.zeros([len(self.inputs), len(self.weights_output_layer)])
        self.errors = np.zeros([len(self.inputs), len(self.weights_output_layer)])
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
        for j in range(1):#range(len(self.inputs)):
            for i in range(len(self.weights_hidden_layer)):
                hidden_layer_induced_fields = np.dot(self.inputs[j][0:3], self.weights_hidden_layer[i]) + self.bias_hidden_layer[i]
                self.hidden_layer_outputs[i] = self.activation_function(hidden_layer_induced_fields)
            
            print(self.hidden_layer_outputs)
            print("\n")
            for i in range(len(self.weights_output_layer)):
                output_layer_induced_fields = np.dot(self.hidden_layer_outputs, self.weights_output_layer[i]) + self.bias_output_layer[i]
                self.outputs[j][i] = self.activation_function(output_layer_induced_fields)
            
            self.errors[j] = self.inputs[j][3:5] - self.outputs[j]
            print(self.outputs[j], self.errors[j])
            


nn = neural_network()
nn.train()
