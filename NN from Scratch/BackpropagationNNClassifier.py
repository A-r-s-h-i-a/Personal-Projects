# Our Neural Net's accuracy is measured by a Cost/Loss Function. This function is a Mean Squared Error (MSE) function as it captures the absolute
# 	error (square), and also captures the amount of values (mean) in its results.
# NOTE: The MSE does NOT represent the proportion of correct predictions!
#
# In order to make the algorithm easier, we can combine the Feedforward portion of the net into the MSE by adding the steps of calculating the
# 	prediction array values before calculating the MSE.
#
# Hin = X*W1 = XW1
# Hout = step(Hin)
# Oin = Hout*W2 = HoutW2
# Oout = step(Oin)
# Y = Actual_Array (ie. Labeled Array, ie. the "Actual" or "True" answers)
#
# MSE = SUM((Output_Array - Actual_Array)^2)/n =   SUM( step( step(XW1) * W2 ) - Y )^2 /n
#
# We want to minimize the Cost Function as it equates to maximizing the accuracy of our NN's predictions. To do this we use the derivative of the MSE.
#	The derivative gives us two important pieces of information, whether the MSE is increasing/decreasing in that direction of the number line (the sign
# 	of the derviative solved at some point), and the amount it will decrease if we move in that direction (the value of the solution). ie. if we
# 	take dY/dw at point p and get -4, we see that if we move further in the +direction on the number line we will decrease the error by a factor of 4.
# 	Our goal is to find the minimum of the derivative, which will appear to be a minima, characterized in part by dY/dw = 0 at the minimum point.
#
# In practice, we find the minimum MSE by first choosing a random weight w, making a prediction, calculating the MSE, and the derivative. From here we
# 	can see what will happen if we were to move in the + direction due to the derivative sign and value. If we see the MSE would increase, we move in
# 	the opposite direction by decreasing w. If we see it would decrease, we do move in that direction and w. To avoid overshoot from poor step-sizes and
# 	highly dynamic functions, we incorporate a scaling factor α (alpha) known as the "learning rate" (usually 0.1, 0.01, or 0.001). We stop when we reach
# 	(or are very close to) a derivative of 0. This is of course done with multiple (not one) randomly chosen weights, predictions, MSEs, and (instead of
# 	a normal derivative) partial derivatives. This entire process is called Gradient Descent!
# W = W - α * ∂MSE(W)/∂W      <---- Gradient Descent
#
# Performing Gradient Descent over a function of functions,	such as our MSE function, is non-trivial. Especially if we are measuring descent with 
# 	respect to a variable that is embedded deeply in the function. For example, if we have F(G(H(x))), finding the partial derivative ∂F/∂x is more 
# 	complicated than the partial ∂G/∂x. To do find either partial, we would first need to formalize	how each layer of functions is affected by the
# 	partial's variable (in this case x). Then we use these formalizations to carry forward the affects of the dynamic variable from the inner functions 
# 	through to the function being measured. Again, take the case of F(G(H(x))). If we want to compute ∂F/∂x, we must first know ∂H/∂x (how x affects 
# 	the innermost function), ∂G/∂H (how the innermost function affects the next function), and ∂F/∂G (how this last function affects our function of 
# 	interest - F). Then, to carry forward the effects of x from the inside of H(x) all the way to F, we would compute:
# 	∂F/∂x = ∂F/∂G * ∂G/∂H * ∂H/∂x     <---- Chain Rule
#	This process is called the Chain Rule! It can also be applied in multivariable situations, ie. you have z(x,y), x(u,v), and y(u,v) then:
#	∂z/∂u = ∂z/∂x * ∂x/∂u + ∂z/∂y * ∂y/∂u
#	∂z/∂v = ∂z/∂x * ∂x/∂v + ∂z/∂y * ∂y/∂v
#	Again, all that is happening is the affect of the dynamic variable is being carried through.
#
# NOTE: For the activation function we cannot use a step function as when x=threshold f'(x)=undefined, and so we use a sigmoid function ie. y=1/(1+e^x). 
#
# Combining all this, we are now ready to understand, create expressions for, and compute ∂MSE/∂W1 and ∂MSE/∂W2. Note that we are performing Gradient
# 	Descent on our MSE function with respect to each weight! This means that as our NN grows, as the number of nodes and thus weights increases, this
# 	will become very computationally intensive! Just this one portion - computing ∂MSE/∂W for each W to minimize the MSE and thus maximize accuracy!
#
# ∂MSE(W1)/∂W1 = ∂MSE/∂Oout * ∂Oout/∂Oin * ∂Oin/∂Hout * ∂Hout/∂Hin * ∂Hin/∂W1
# ∂MSE(W2)/∂W2 = ∂MSE/∂Oout * ∂Oout/∂Oin * ∂Oin/∂W2
#
# We can see that our partial terms are with respect to entire matrices, not scalars. Numpy does these computations via Matrix Calculus, and if we were
# 	using the library for this program we would stop here. However we are not, and so must break the problem down further. By diving into the matrices
# 	and analyzing their expressions, we see that when we take the partial derivative for the latter equation (∂MSE(W2)/∂W2) it reduces to:
#
# ∂MSE(W2)/∂W2 = 1/N * SUMe((ooutn^e-yn^e)*(ooutn^e*(1-ooutn^e)*houtn^e))
#
# This corresponds to its Chain Rule partial derivative terms as such:
# 
# (ooutn^e-yn^e) = ∂MSE/∂Oout
# (ooutn^e*(1-ooutn^e) = ∂Oout/∂Oin
# houtn^e = ∂Oin/∂W2
#
# At the Matrix-scale, this is written:
#
# Oerror = Oout - Y     where Oout is the output of the NN, and Y the labeled data, and so their difference is the error of the NN's output
# Odelta = Oerror (*) Oout (*) (J - Oout)     where (*) is element wise multiplication, and J is a ones matrix (standard notation)
# W2update = 1/N * (Hout^T * Odelta)     where W2update is the weight update matrix, Hout the output of the hidden layer, and "^T" is a transpose
#
# For the former case, we have the following end Matrix-scale Chain Rule partial derivative terms (note that there are overlapping partial derivative 
# 	in the Chain Rule terms, and so there will be in the results):
#
# Oerror = Oout - Y
# Odelta = Oerror (*) Oout (*) (J - Oout)
# Herror = Odelta * W2^T
# Hdelta = Herror (*) Hout (*) (J - Hout)     where Hout is the output of the hidden layer
# W1update = 1/N * (X^T * Hdelta)
#
# Note that in the Backpropagation algorithm, it is only AFTER calculating the MSE and thus the update matrices for the weights that the Gradient Descent
# 	step is performed! Thus the full process (Feedforward and Backpropagation together) is as follows:
# 1) Multiply the inputs and resulting values by weights and sum them appropriately through the input and hidden layers, all the way to the output 
# 		(entirety of the Feedforward algorithm).
# 2) Compare the NN's output to the labeled output for those inputs of this test datapoint(s), in other words Oout - Y (Backpropagation begins).
# 3) If unsatisfied with the accuracy of the model, begin Gradient Descent by calculating the partial derivative of the MSE of the NN's output with 
# 		respect to each weight.
# 4) Compute the update matrices for each weight and update the weight values using some appropriately selected learning rate.
# 5) Return to step 1 with the new weights and preferrably new test datapoint(s).
#
# A few notes on training. If the learning rate, α, is too large, you may end up never reaching an MSE as low as desired (or your MSE may actually grow). 
# 	Too small and the number of iterations necessary to minimize the MSE will proportionally increase requiring greater compute for training. 
#	If you find the MSE never decreases as much as desired, a larger Neural Network may be required. In other words, the Neural Net being used is too
#	small/simple to learn the patterns in the data. In that case you could increase the number of neurons or layers. Playing with these features and 
#	looking at these indicators (MSE, training time/compute) are a large part of training.

import pandas as pd
import numpy as np #ONLY FOR LINEAR ALGEBRA!
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

############################################################################################################################
##################################################### HELPER FUNCTIONS #####################################################
############################################################################################################################

# SIGMOID FUNCTION ---------------------------------------------------------------------------------------------------------
# A sigmoid is a smooth activation-like function differentiable at every point. It returns 0 for every input<0 and 1 for every
# 	input>1. During [0,1] it returns intermediary values (the cost of being continuously differentiable).
def sigmoid(x):
	return 1/(1+np.exp(-x))
# --------------------------------------------------------------------------------------------------------------------------

# MEAN SQUARED ERROR FUNCTION ----------------------------------------------------------------------------------------------
# For some predictions and their corresponding labels, this function calculates and returns their mean squared error value.
def mean_squared_error(predictions, labels):
	N = labels.size
	mse = ((predictions - labels)**2).sum() / (2*N)
	return mse
# --------------------------------------------------------------------------------------------------------------------------

# ACCURACY FUNCTION --------------------------------------------------------------------------------------------------------
# For some predictions and their corresponding labels, this function calculates and returns the prediction's accuracy.
def accuracy(predictions, labels):
	predictions_correct = predictions.argmax(axis=1) == labels.argmax(axis=1) #creates list of booleans for 1:1 prediction-label comparisons
	accuracy = predictions_correct.mean() #the mean ends up being the accuracy as True=1 and False=0
	return accuracy
# --------------------------------------------------------------------------------------------------------------------------

############################################################################################################################
############################################################################################################################
############################################################################################################################



if __name__ == "__main__":

	# LOAD/EDIT IRIS FLOWER DATASET --------------------------------------------------------------------------------------------
	print("Loading Data")
	df = pd.read_csv("Iris.csv")
	# INPUT (X) AND LABEL (Y) MATRICES -----------------------------------------------------------------------------------------
	y = pd.get_dummies(df.species).values
	x = df.drop(["Id", "species"], axis=1).values
	# TEST AND TRAIN DATA SPLIT ------------------------------------------------------------------------------------------------
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=4) #setting a random state ensures the same random split
	# --------------------------------------------------------------------------------------------------------------------------



	# INITIALIZE PARAMETERS ----------------------------------------------------------------------------------------------------
	print("Initializing NN Parameters")
	learning_rate = 0.1 #learning scaling factor, used in Gradient Descent formula for updating weights
	epochs = 30000 #training iterations
	N = y.size #the number of elements in the label matrix Y and thus is also the number of elements in the NN's output matrix
	# The number of nodes in each layer
	n_input = 4
	n_hidden = 2
	n_output = 3
	# Initializing weights randomly
	np.random.seed(10)
	weights_1 = np.random.normal(scale=0.5, size=(n_input, n_hidden))  # n_input x n_hidden size matrix with random values at each location normally distributed with a standard deviation of 0.5
	weights_2 = np.random.normal(scale=0.5, size=(n_hidden, n_output)) # n_hidden x n_output size matrix, rest same as above
	# Initializing dictionary to track the MSE and accuracy of the NN
	monitoring = {"mean_squared_error": [], "accuracy": []}
	# --------------------------------------------------------------------------------------------------------------------------



	# TRAIN NEURAL NETWORK -----------------------------------------------------------------------------------------------------
	print("Training...")
	for epoch in range(epochs):

		# FEEDFORWARD
		# Inputs through hidden layer (w1)
		hidden_layer_inputs = np.dot(x_train, weights_1)
		hidden_layer_outputs = sigmoid(hidden_layer_inputs)
		# Hidden layer values through output layer (w2)
		output_layer_inputs = np.dot(hidden_layer_outputs, weights_2)
		output_layer_outputs = sigmoid(output_layer_inputs)
		
		# Monitor and Record Training Process
		mse = mean_squared_error(output_layer_outputs, y_train)
		acc = accuracy(output_layer_outputs, y_train)
		monitoring["mean_squared_error"].append(mse)
		monitoring["accuracy"].append(acc)

		# BACKPROPAGATION
		# Calculate partial sub-terms for Gradient Descent
		output_layer_error = output_layer_outputs - y_train
		output_layer_delta = output_layer_error * output_layer_outputs * (1 - output_layer_outputs)
		hidden_layer_error = np.dot(output_layer_delta, weights_2.T)
		hidden_layer_delta = hidden_layer_error * hidden_layer_outputs * (1 - hidden_layer_outputs)
		# UPDATE WEIGHTS
		# Calculate the update matrices
		weights_2_update = np.dot(hidden_layer_outputs.T, output_layer_delta) / N
		weights_1_update = np.dot(x_train.T, hidden_layer_delta) / N
		# The Gradient Descent Formula Step!
		weights_2 = weights_2 - learning_rate * weights_2_update
		weights_1 = weights_1 - learning_rate * weights_1_update

	monitoring_df = pd.DataFrame(monitoring)
	print("Training Complete")
	# --------------------------------------------------------------------------------------------------------------------------



	# VISUALIZE LEARNING -------------------------------------------------------------------------------------------------------
	sns.set_style("darkgrid")
	fig, axes = plt.subplots(1, 2, figsize=(13, 5))
	monitoring_df.mean_squared_error.plot(ax=axes[0], title="Mean Squared Error")
	monitoring_df.accuracy.plot(ax=axes[1], title="Accuracy")
	plt.show()
	# --------------------------------------------------------------------------------------------------------------------------



	# TEST DATA ----------------------------------------------------------------------------------------------------------------
	# Test data as inputs through hidden layer (w1)
	hidden_layer_inputs = np.dot(x_test, weights_1) #NOTE: Now using test data x_test!
	hidden_layer_outputs = sigmoid(hidden_layer_inputs)
	# Hidden layer values through output layer (w2)
	output_layer_inputs = np.dot(hidden_layer_outputs, weights_2)
	output_layer_outputs = sigmoid(output_layer_inputs)

	# Analyze Output
	acc = accuracy(output_layer_outputs, y_test) #NOTE: Comparing outputs to y_test!
	print("\nModel Accuracy: {}".format(acc),"\n")
	# --------------------------------------------------------------------------------------------------------------------------


