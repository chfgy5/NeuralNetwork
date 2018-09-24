import numpy
import csv

inputs = []
bias_hidden_layer = []
weights_hidden_layer = []
bias_output_layer = []
weights_output_layer = []

with open("./inputs/cross_data (3 inputs - 2 outputs).csv") as input_csv:
    csv_reader = csv.reader(input_csv, delimiter=',')
    for row in csv_reader:
        arr = []
        for item in row:
            arr.append(float(item))
        inputs.append(arr)

with open("./inputs/b1 (11 nodes).csv") as bias_hidden_layer_csv:
    csv_reader = csv.reader(bias_hidden_layer_csv, delimiter=',')
    for row in csv_reader:
        arr = []
        for item in row:
            arr.append(float(item))
        bias_hidden_layer.append(arr)

with open("./inputs/w1 (3 inputs - 11 nodes).csv") as input_weights_hidden_layer_csv:
    csv_reader = csv.reader(input_weights_hidden_layer_csv, delimiter=',')
    for row in csv_reader:
        arr = []
        for item in row:
            arr.append(float(item))
        weights_hidden_layer.append(arr)

with open("./inputs/b2 (2 output nodes).csv") as bias_output_layer_csv:
    csv_reader = csv.reader(bias_output_layer_csv, delimiter=',')
    for row in csv_reader:
        arr = []
        for item in row:
            arr.append(float(item))
        bias_output_layer.append(arr)

with open("./inputs/w2 (from 11 to 2).csv") as input_weights_output_layer_csv:
    csv_reader = csv.reader(input_weights_output_layer_csv, delimiter=',')
    for row in csv_reader:
        arr = []
        for item in row:
            arr.append(float(item))
        weights_output_layer.append(arr)

print("inputs: ", inputs)
print("bias hidden: ", bias_hidden_layer)
print("weights hidden: ", weights_hidden_layer)
print("bias output: ", bias_output_layer)
print("weights output: ", weights_output_layer)
print("\n\n")