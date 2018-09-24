import numpy as np
import csv

def sigmoid(x):
    return 1 / (1 + np.e ** -x)

class neural_network:
    def __init__(self):
        self.inputs = np.array(self.get_given_values("./inputs/cross_data (3 inputs - 2 outputs).csv"))
        self.bias_hidden_layer = np.array(self.get_given_values("./inputs/b1 (11 nodes).csv"))
        self.weights_hidden_layer = np.array(self.get_given_values("./inputs/w1 (3 inputs - 11 nodes).csv"))
        self.hidden_layer_outputs = np.zeros(len(self.weights_hidden_layer))
        self.bias_output_layer = np.array(self.get_given_values("./inputs/b2 (2 output nodes).csv"))
        self.weights_output_layer = np.array(self.get_given_values("./inputs/w2 (from 11 to 2).csv"))
        self.outputs = np.zeros([len(self.inputs), len(self.weights_output_layer)])
        self.errors = np.zeros([len(self.inputs), len(self.weights_output_layer)])
        self.delta_output = np.zeros([len(self.inputs), len(self.weights_output_layer)])
        self.delta_input = np.zeros([len(self.inputs), len(self.weights_hidden_layer)])
        self.delta_weights_hidden = np.zeros([len(self.weights_hidden_layer), len(self.weights_hidden_layer[0])])
        self.delta_weights_output = np.zeros([len(self.weights_output_layer), len(self.weights_output_layer[0])])
        self.activation_function = sigmoid
        self.learning_rate = .7
        self.momentum = .3

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
            
            for i in range(len(self.weights_output_layer)):
                output_layer_induced_fields = np.dot(self.hidden_layer_outputs, self.weights_output_layer[i]) + self.bias_output_layer[i]
                self.outputs[j][i] = self.activation_function(output_layer_induced_fields)
            
            self.errors[j] = self.inputs[j][3:5] - self.outputs[j]
            
            #start backprop
            self.delta_output[j] = self.errors[j] * self.outputs[j] * (1 - self.outputs[j])
            self.delta_input[j] = self.hidden_layer_outputs * (1 - self.hidden_layer_outputs) * np.dot(self.delta_output[j], self.weights_output_layer)
                     
            #Not using momentum becuase delta w (k-1) not given
            for l in range(len(self.delta_output[j])):
                self.delta_weights_output = self.learning_rate * self.delta_output[j][l] * self.hidden_layer_outputs
            for l in range(len(self.delta_input[j])):
                self.delta_weights_hidden[l] = self.learning_rate * self.delta_input[j][l] * self.inputs[j][0:3]

            self.weights_hidden_layer = self.weights_hidden_layer - self.delta_weights_hidden
            self.weights_output_layer = self.weights_output_layer - self.delta_weights_output



nn = neural_network()
nn.train()

