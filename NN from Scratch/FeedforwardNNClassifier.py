# Feedforward Neural Net from Scratch! There is no learning happening here, this is just the structure of the NN - ie. how the algo makes a decision!
# A Neural Net is a machine learning algorithm loosely inspired by the human brain. It is composed of many Artificial Neurons organized into layers.
# An Artifical Neuron takes in an input "x", multiplies it by some biasing weight "w", then feeds the result into an activation (step) function. The 
# 	output of one neuron from one layer is fed into many neurons from the next layer. The very first layer of neurons which interact with the outside 
# 	world is called the input layer, the very last layer the output layer, and the layers in-between are called hidden layers. Depending on the values
# 	fed to the input layer, different neurons will fire in the hidden layers and a different output will emerge from the network. Training the network 
#	determines the weights "w" for all the different neurons.
#
# Linear Algebra (vectorization) is used in deep learning due to its ability to significantly optimize algorithms (by orders of magnitude). As a result,
# 	it is important that information (inputs, weights, etc.) be stored in arrays formatted properly so that mathematically operations are succesful.
#
# For this NN from scratch, we will be building a classifier for the Iris Flower Dataset. The NN will have no hidden layers, just a 2 neuron input layer
#	taking 4 data inputs, and a 3 neuron output layer with 3 1-hot outputs (ach 1-hot indicating one of the 3 flowers in the dataset).
#
# An artificial neuron takes in inputs (x), multiplies them each by some weight (w), then sums the products (x*w). If the sum is greater than some
# 	threshold value T, its activation function outputs a value 1 (otherwise a 0 is output).

from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")



# WEIGHTED SUM FUNCTION ----------------------------------------------------------------------------------------------------
# This is essentially the dot product...if we use numpy (linear algebra) it'll be faster...
def weighted_sum(inputs, weights):
	total = 0
	#for each item in list (must be same size), multiply and add to total!
	for input_value, weight in zip(inputs, weights):
		total += input_value * weight

	return total
# --------------------------------------------------------------------------------------------------------------------------



# STEP FUNCTION ------------------------------------------------------------------------------------------------------------
# VERY SIMPLE STEP FUNCTION - Analyzes a single input and returns a single output based on a threshold (default 2)
# def step_function(number, threshold=2):
# 	if number >= threshold:
# 		return 1
# 	else:
# 		return 0
# --------------------------------------------------------------------------------------------------------------------------



# STEP FUNCTION ------------------------------------------------------------------------------------------------------------
# MORE COMPLEX STEP FUNCTION - Analyzes an input array and returns an output array where each value is 1 if that input value was
# 	greater than the threshold (default 2) and 0 otherwise.
def step_function(input_array, threshold=2):
	
	return (input_array >= threshold).astype(int)
# --------------------------------------------------------------------------------------------------------------------------



# DETERMINE LAYER OUTPUTS FUNCTION -----------------------------------------------------------------------------------------
# Each input is a list of lists, with columns referring to inputs, and rows referring to dataset examples.
def determine_layer_outputs(list_of_inputs, list_of_weights, activation_function=True):

	layer_outputs = []
	for inputs in list_of_inputs: #iterate through each row, each example fed to the neuron as an input, a list itself
		node_outputs = [] #temporary sublist for this iteration to contain the layer outputs
		for weights in list_of_weights: #iterate through each row of weights
			node_input = weighted_sum(inputs, weights) #take the weighted sum of the current inputs list with this weights list
			if activation_function: #next, see if each neuron fires by checking the activation function against this sum
				node_output = step_function(node_input)
			else:
				node_output = node_input #this (and the activation_function input) lets us see the hidden_layer weighed values 
			node_outputs.append(node_output) #pass along the neuron outputs to the layer outputs
		layer_outputs.append(node_outputs) #add the sublist to the output list, building an output list of lists to pass on

	return layer_outputs
# --------------------------------------------------------------------------------------------------------------------------



if __name__ == '__main__':

	# LOAD/EDIT IRIS FLOWER DATASET --------------------------------------------------------------------------------------------
	df = pd.read_csv("Iris.csv")
	# one-hot encoding
	species_one_hot = pd.get_dummies(df.species)
	df = df.join(species_one_hot)
	df = df.drop(["Id", "species"], axis=1)
	# --------------------------------------------------------------------------------------------------------------------------


	inputs = np.array([[ 4.9, 3.0, 1.4, 0.2],
			  		   [ 6.4, 3.2, 4.5, 1.5],
			  		   [ 5.8, 2.7, 5.1, 1.9]])
	weights1 = np.array([[ 0.9,-0.5],
				[ 0.8,-0.5],
				[-1.0, 1.5],
				[-1.0, 1.0]])
	weights2 = np.array([[ 2, 1,-1],
						 [-1, 1, 2]])



	hidden_layer_inputs = np.dot(inputs, weights1) #Instead of using the weighted sum function, just taking the dot product of the inputs and weights
	print(hidden_layer_inputs)
	hidden_layer_outputs = step_function(hidden_layer_inputs)
	print(hidden_layer_outputs)
	output_layer_inputs = np.dot(hidden_layer_outputs,weights2) #Instead of using the weighted sum function, just taking the dot product of the inputs and weights
	print(output_layer_inputs)
	output_layer_outputs = step_function(output_layer_inputs)
	print(output_layer_outputs)
