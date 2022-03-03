# Neural Network without PyTorch/TensorFlow/Keras to classify passengers on the "Titanic" data set. The classification being whether 
# 	they surivived or not...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



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
	predictions_correct = predictions.round() == labels #creates list of booleans for 1:1 prediction-label comparisons
	accuracy = predictions_correct.mean() #the mean ends up being the accuracy as True=1 and False=0
	return accuracy
# --------------------------------------------------------------------------------------------------------------------------

############################################################################################################################
############################################################################################################################
############################################################################################################################



if __name__ == '__main__':

	# PREPARE DATA -------------------------------------------------------------------------------------------------------------
	# Import and drop unnecessary columns
	print("Preparing Data")
	df = pd.read_csv("Titanic.csv")
	df = df.drop(["PassengerID", "Name", "Ticket", "Cabin"], axis=1)
	# Fill in missing values
	df = df.fillna({"Age": df.Age.median(), "Embarked": df.Embarked.mode()[0]})
	# One-Hot encoding
	df = pd.get_dummies(df, columns=["Pclass", "Sex", "SibSp", "Parch", "Embarked"])
	#Min-Max scaling
	scaler = MinMaxScaler()
	df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
	# print(df.head())
	# --------------------------------------------------------------------------------------------------------------------------



	# TRAIN TEST SPLIT ---------------------------------------------------------------------------------------------------------
	x = df.drop("Survived", axis=1).values
	y = df[["Survived"]].values
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) #setting a random state ensures the same random split
	# --------------------------------------------------------------------------------------------------------------------------



	# SETUP PARAMETERS ---------------------------------------------------------------------------------------------------------
	print("Initializing NN Parameters")
	learning_rate = 0.1 #the scaling factor used in Gradient Descent to update the weights by, poorly chosen and it will break the program
	epochs = 10000 #training iterations, the "compute"
	N = y_train.size #number of labels
	n_input = 24 #number of input layer nodes
	n_hidden = 4 #number of hidden layer nodes, only one hidden layer!
	n_output = 1 #number of output layer nodes (1-hot, yes or no, survived or died)
	np.random.seed(10) #allows us to control for the randomness so that initial random weight values are the same program-start to program-start
	weights_1 = np.random.normal(scale=0.5, size=(n_input, n_hidden)) # n_input x n_hidden size matrix with random values at each location normally distributed with a standard deviation of 0.5
	weights_2 = np.random.normal(scale=0.5, size=(n_hidden, n_output)) # n_hidden x n_output size matrix, rest same as above
	monitoring = {"mean_squared_error": [], "accuracy": []} # dictionary to track the MSE and accuracy of the model during trainging
	# --------------------------------------------------------------------------------------------------------------------------



	# TRAIN NEURAL NETWORK -----------------------------------------------------------------------------------------------------
	print("Training NN...")
	for epoch in range(epochs):

		# FEEDFORWARD
		hidden_layer = sigmoid(np.dot(x_train, weights_1)) #inputs through hidden layer
		output_layer = sigmoid(np.dot(hidden_layer, weights_2)) #inputs through output layer

		# MONITOR/RECORD PERFORMANCE
		mse = mean_squared_error(output_layer, y_train)
		acc = accuracy(output_layer, y_train)
		monitoring["mean_squared_error"].append(mse)
		monitoring["accuracy"].append(acc)

		# BACKPROPAGATION
		# Calculate partial derivative subterms to measure MSE change with respect to each weight
		output_delta = (output_layer - y_train) * output_layer * (1 - output_layer) # Error changes relative to output layer
		hidden_delta = np.dot(output_delta, weights_2.T) * hidden_layer * (1 - hidden_layer) # Error changes relative to hidden layer
		# Update Weights - Here we use the Gradient Descent formula and Learning Rate
		weights_2 -= learning_rate * np.dot(hidden_layer.T, output_delta) / N
		weights_1 -= learning_rate * np.dot(x_train.T, hidden_delta) / N

	print("Training Complete")
	monitoring_df = pd.DataFrame(monitoring) # transform monitoring dictionary into a Pandas DataFrame
	# --------------------------------------------------------------------------------------------------------------------------



	# PLOT TRAINING PERFORMANCE ------------------------------------------------------------------------------------------------
	sns.set_style("darkgrid")
	fig, axes = plt.subplots(1, 2, figsize=(13, 5))
	monitoring_df.mean_squared_error.plot(ax=axes[0], title="Mean Squared Error")
	monitoring_df.accuracy.plot(ax=axes[1], title="Accuracy")
	plt.show()
	# --------------------------------------------------------------------------------------------------------------------------



	# TEST MODEL ---------------------------------------------------------------------------------------------------------------
	# Essentially the same code as the feedforward part of the NN training loop, but with the test data
	hidden_layer = sigmoid(np.dot(x_test, weights_1)) #test data through hidden layer
	output_layer = sigmoid(np.dot(hidden_layer, weights_2)) #test data through output layer
	acc = accuracy(output_layer, y_test) # measure accuracy
	print(acc)
	# --------------------------------------------------------------------------------------------------------------------------


