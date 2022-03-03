# Random Forest Algorithm from scratch!
# A Random Forest feeds data to many Decision Trees in parallel, then analyzes the Tree's outputs to come to a decision/overall-output.
# By using many Decision Trees, it minimizes potential issues/errors that may occur from just one. 

# We want to create many Decision Trees in our Forest from the same training dataset that are similar, but not exactly the same.
# To do this we must introduce some randomness into the dataset. One way we can do this is with a technique called "Bootstrapping".
# Bootstrapping is a statistical resampling technique that involves random sampling of a dataset WITH REPLACEMENT. 
# For example, while the original dataset will have datapoints 1, 2, 3, 4, 5 - the bootstrapped dataset might have datapoints 3, 4, 3, 5, 1.
# So, we create multiple bootstrapped datasets from the original training data, then create decision trees for each bootstrapped dataset!
# The second way we can introduce randomness into the data is by introducing it into the features with a technique called the "Random Subspace Method".
# The idea here is that when we create nodes in a tree, instead of considering all the data's features, we simply consider a random subspace of them.
# Combining the Random Subspace Method with Bootstrapping, we can create as many random trees as we want from the same training data!
# With those tools available, the next step is to generate our Forest of Decision Trees. Then feed those Trees our data and measure their outputs.
# Finally, the outputs of the Trees are tallied like votes in the case of classification. Whichever class recieves the most votes is the output of
# 	the overall Random Forest Algorithm!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint
from DecisionTreeAlgorithm import decision_tree_algorithm, decision_tree_predictions, predict_example, train_test_split



# CALCULATE ACCURACTY ------------------------------------------------------------------------------------------------------
# This function calculates the accuracty of some predictions by comparing them to the actual answers (labels) they were ment to predict.
def calculate_accuracy(predictions, labels):
	predictions_correct = (predictions==labels) #Boolean array created, True for where prediction==Label, False otherwise
	accuracy = predictions_correct.mean() #Mean of the array is the accuracy

	return accuracy
# --------------------------------------------------------------------------------------------------------------------------



# BOOTSTRAPPING ------------------------------------------------------------------------------------------------------------
# This function creates a bootstrapped dataset, that is, it introduces randomness into some "original" dataset in order to create a Random Forest.
# It does this by randomly sampling from the dataset WITH REPLACEMENT! The number of times it samples (size of the new dataset) is an input.
def bootstrapping(train_df, n_bootstrap):
	
	#Creating a 1D array of ints with length n_bootstrap, containing random integer values <= the size of the training dataset to act as indices
	#	for the randomly sampled (bootstrapped) dataset
	bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
	#Create bootstrapped dataset by using those indices to choose datapoints
	df_bootstrapped = train_df.iloc[bootstrap_indices]

	return df_bootstrapped
# --------------------------------------------------------------------------------------------------------------------------



# RANDOM FOREST ALGORITHM --------------------------------------------------------------------------------------------------
# This is the main algorithm of the project, it spawns multiple Decision Trees and trains them each using bootstrapped data and a subspace of the
# 	available features in the training data (each one uniquely).
# As inputs it takes in the original training dataset, desired number of Decision Trees, desired number of datapoints in the bootstrapped dataset,
#	desired number of features in the feature subspace, and the desired number of layers in the Decision Trees.
# The datatype will be as a list, containing each tree as an item (trees are dictionaries)
def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
	
	forest = []
	for i in range(n_trees):
		df_bootstrapped = bootstrapping(train_df, n_bootstrap) #Create a unique dataset
		tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features) #Train a tree with it a subspace of features
		forest.append(tree) #Add the tree to the forest

	return forest
# --------------------------------------------------------------------------------------------------------------------------



# RANDOM FOREST PREDICTIONS -------------------------------------------------------------------------------------------------
# Create the Random Forest's predictions by having the Decision Trees make predictions for each test datapoint, then choosing the most
# common prediction for each datapoint as the Forest's prediction for that point.
def random_forest_predictions(test_df, forest):

	df_predictions = {} #the forest of predictions will be held in a dictionary where each tree's name is the key and its prediction the value

	for i in range(len(forest)): #loop through each tree, adding it and its 1D list of predictions (from the test data) to the dictionary
		column_name = "tree_{}".format(i) #create a variable containing the text of the tree name currently being examined
		predictions = decision_tree_predictions(test_df, tree=forest[i]) #generate a prediction list from the test list for the current tree 
		df_predictions[column_name] = predictions #create the dictionary entry for the current tree & its predictions

	df_predictions = pd.DataFrame(df_predictions) # transform the dictionary into a dataframe!

	# choose the value that appears the most often as the Forest's prediction, each column is a tree, each row a test datapoint, so in effect
	#	we are choosing the average bool from each row as the prediction for that row's datapoint, this can be done with the following function
	random_forest_predictions = df_predictions.mode(axis=1)[0] # This array now contains the Random Forest's predictions for each test datapoint

	return random_forest_predictions
# --------------------------------------------------------------------------------------------------------------------------



if __name__ == '__main__':
	
	# LOAD AND PREPARE WINE DATA SET -------------------------------------------------------------------------------------------
	df = pd.read_csv("winequality-red.csv") #Import csv data
	df["label"] = df.quality
	df = df.drop("quality", axis=1) #Rename "quality" column to column named "label"
	# Ensure the contents of a column's cell has NO WHITESPACE, otherwise DecisionTreeAlgorithm.py won't parse it correctly!!!!!!
	column_names=[]
	for column in df.columns:
		name = column.replace(" ", "_")
		column_names.append(name)
	df.columns = column_names
	def transform_label(value): #label all wine scored <= 5 "bad" wine, and >5 "good" wine 
		if value <= 5:
			return "bad"
		else:
			return "good"
	df["label"] = df.label.apply(transform_label) #apply the function transform_label to score all the wine in the imported wine dataset
	wine_quality = df.label.value_counts(normalize=True) #create the two groups of "bad" and "good" wine as percentages 
	print(wine_quality) #print the wine_quality values
	wine_quality[["bad", "good"]].plot(kind="bar",color=["blue","orange"]) #plot the group data in a bar graph
	plt.show()
	# --------------------------------------------------------------------------------------------------------------------------



	# MAIN RUN -----------------------------------------------------------------------------------------------------------------
	random.seed(0)
	train_df, test_df = train_test_split(df, test_size=0.2)

	forest = random_forest_algorithm(train_df, n_trees=5, n_bootstrap=500, n_features=4, dt_max_depth=4)
	predictions = random_forest_predictions(test_df, forest)
	accuracy = calculate_accuracy(predictions, test_df.label)
	print(accuracy)
	# --------------------------------------------------------------------------------------------------------------------------



	# COMPARING RANDOM FOREST ALGORITHM TO ONE DECISION TREE -------------------------------------------------------------------
	# random.seed(0)
	# train_df, test_df = train_test_split(df, test_size=0.2)
	# accuracies = []

	# for i in range(100):
	# 	forest = random_forest_algorithm(train_df, n_trees=10, n_bootstrap=800, n_features=4, dt_max_depth=4)
	# 	predictions = random_forest_predictions(test_df, forest)
	# 	accuracy = calculate_accuracy(predictions, test_df.label)
	# 	print("RFA Accuracy: ", accuracy, "\t", i)
	# 	accuracies.append(accuracy)

	# print("Average accuracy of the Random Forest Algorithm: {}".format(np.array(accuracies).mean()))
	# accuracies = []

	# for i in range(100):
	# 	forest = random_forest_algorithm(train_df, n_trees=1, n_bootstrap=len(train_df), n_features=999, dt_max_depth=4)
	# 	predictions = random_forest_predictions(test_df, forest)
	# 	accuracy = calculate_accuracy(predictions, test_df.label)
	# 	print("DT Accuracy: ", accuracy, "\t", i)
	# 	accuracies.append(accuracy)

	# print("Average accuracy of the Decision Tree Algorithm: {}".format(np.array(accuracies).mean()))
	# --------------------------------------------------------------------------------------------------------------------------



	# COMPARING MY RANDOM FOREST ALGORITHM TO SCKIT'S --------------------------------------------------------------------------
	# # Load, prepare, and run iris scikit data/random forest algorithm (datacamp.com/community/tutorials/random-forests-classifier-python)
	# # Over 1000 runs, SKLearn seems to perform only 5.04% better! However, SKLearns algorithm took 20.7 seconds to execute 1000 times, while
	# # 	mine took 970.6 seconds.
	# from sklearn import datasets as skl_datasets
	# from sklearn import metrics as skl_metrics
	# from sklearn.model_selection import train_test_split as skl_train_test_split
	# from sklearn.ensemble import RandomForestClassifier as skl_RandomForestClassifier
	# iris = skl_datasets.load_iris()
	# df = pd.DataFrame({
	# 	'sepal length':iris.data[:,0],
	# 	'sepal width':iris.data[:,1],
	# 	'petal length':iris.data[:,2],
	# 	'petal width':iris.data[:,3],
	# 	'species':iris.target
	# 	})
	# skl_df = df
	# skl_X = skl_df[skl_df.columns.values[:(len(skl_df.columns.values)-1)]]
	# skl_Y = skl_df[skl_df.columns.values[len(skl_df.columns.values)-1]]
	# skl_X_train, skl_X_test, skl_Y_train, skl_Y_test = skl_train_test_split(skl_X, skl_Y, test_size=0.2)
	# clf = skl_RandomForestClassifier(n_estimators=10) #chosen to match the parameters of my algorithm, a large constraint on sklearn's algorithm
	# skl_accuracies = []
	# for i in range(1000):
	# 	print(i)
	# 	clf.fit(skl_X_train, skl_Y_train)
	# 	skl_Y_pred = clf.predict(skl_X_test)
	# 	skl_accuracies.append(skl_metrics.accuracy_score(skl_Y_test, skl_Y_pred))
	# print("SKLearn's Random Forest Algorithm Average Accuracy {}".format(np.array(skl_accuracies).mean(), "\n"))

	# # Run my Random Forest Algorithm for the same dataset
	# column_names=[]
	# for column in df.columns: #ensure there is no whitespace in the column names so that the decision tree algorithm questions dont break
	# 	name = column.replace(" ", "_") 
	# 	column_names.append(name)
	# df.columns = column_names
	# df["label"] = df.species
	# df = df.drop("species", axis=1) #Rename "label" column to column named "quality"
	# train_df, test_df = train_test_split(df, test_size=0.2)
	# accuracies = []
	# for i in range(1000):
	# 	print(i)
	# 	forest = random_forest_algorithm(train_df, n_trees=10, n_bootstrap=1000, n_features=4, dt_max_depth=4)
	# 	predictions = random_forest_predictions(test_df, forest)
	# 	accuracy = calculate_accuracy(predictions, test_df.label)
	# 	accuracies.append(accuracy)
	# print("Arshia's Random Forest Algorithm Average Accuracy {}".format(np.array(accuracies).mean(), "\n"))
	# --------------------------------------------------------------------------------------------------------------------------

