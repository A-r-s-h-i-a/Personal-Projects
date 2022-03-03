# Decision Tree Algorithm from scratch!
# A Decision Tree splits data by features through a series of questions.
# A question is asked at each node/leaf, the answer to which takes an agent along one of two branches/paths.

# The way that an algorithm knows what question to ask in order to split a dataset is by looking at the uncertainty or entropy of the data.
# It works to choose a value by which to split the data that maximizes the reduction in entropy of the resulting inhomogenous dataset.
# Possible splitting values analyzed are only those equidistant between two datapoints.

# So the algorithm looks at each splitting value, calculates the classification-specific and overall entropy from each split, then chooses splits with 
# 	the least entropy! It does this until each classification has been split down branches leading to 0% entropy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint



# TRAIN-TEST-SPLIT ---------------------------------------------------------------------------------------------------------
# This function takes our data set, splits it into training and test datasets, then returns them.
def train_test_split(df, test_size):

	indices = df.index.tolist() #The indices for each datapoint from the dataset, in a list 
	
	if (test_size<1 and test_size>0): #If a percentage is given for test_size, convert this to a number of samples
		test_size = round(test_size*len(indices))

	test_indices = random.sample(population=indices, k=test_size) #Randomly sample from the dataset the relevant number of times
	test_df = df.loc[test_indices] #Capture test datapoints in one list, training datapoints in another, ensuring they are unique
	train_df = df.drop(test_indices)

	return train_df, test_df
# --------------------------------------------------------------------------------------------------------------------------



# DATA PURE CHECK ----------------------------------------------------------------------------------------------------------
# This function checks if a subpartition of our data contains only one class, is "pure".
def check_purity(data):

	label_column = data[:,-1] #The column containing all of the datapoint labels
	unique_classes = np.unique(label_column) #Tells us how many unique values there are, thus how many unique classifications

	if (len(unique_classes) == 1): #If there is one class, it is pure! Else, it isn't pure.
		return True
	else:
		return False
# --------------------------------------------------------------------------------------------------------------------------



# CLASSIFY -----------------------------------------------------------------------------------------------------------------
# This function classifies the data, it does this by choosing the classification that appears most often in the now-determined "pure" dataset.
def classify_data(data):
	
	label_column = data[:,-1] #The column containing all of the datapoint labels
	unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True) 
	#Returns the number of times each existing label appears, ie. votes for each
	#Whichever classification has the most counts/votes is the one chosen as the classification!
	index = counts_unique_classes.argmax() #The classification with the most counts is the index
	classification = unique_classes[index]

	return classification
# --------------------------------------------------------------------------------------------------------------------------



# POTENTIAL SPLITS ---------------------------------------------------------------------------------------------------------
# Determine where the potential split-points of the dataset are, returning these points as a dictionary.
# The keys to this dictionary will be the indices to a datatype's column in the data.
# For example, a key "n" refers to a specific datatype (column in data) and all the corresponding values are split points for that datatype.
# random_subspace added for the Random Forest Algorithm project, not part of this program/project - ignore it if focusing only Decision Trees.
def get_potential_splits(data, random_subspace):

	potential_splits = {}
	n_rows, n_columns = data.shape #Capture the number of columns in the data
	column_indices = list(range(n_columns - 1)) #Creates a list that goes from 0 to the number of columns we have

	if random_subspace and random_subspace <= len(column_indices): #If the Random Forest Algorithm is using this function (and attempting to use only a subset of the features)
		column_indices = random.sample(population=column_indices, k=random_subspace)

	for column_index in column_indices: #For each column/feature we are looking at
		potential_splits[column_index] = [] #Create list of potential splits
		values = data[:,column_index] #The values from the column we are "looking" at in this loop iteration in list form
		unique_values = np.unique(values) #Filter values
		potential_splits[column_index] = unique_values

	return potential_splits 
# --------------------------------------------------------------------------------------------------------------------------



# SPLIT DATA ---------------------------------------------------------------------------------------------------------------
# This function splits a sorted list of data at a certain datapoint, returning two lists - the data above and below this point. 
def split_data(data, split_column, split_value):

	# 1) Find the datapoint in the list (depending on the numpy functions available, be putting all the datapoints in a list as you iterate)
	# 2) Put all the points below in the list data_below, sorted. Then put all the others in a list data_above, also sorted.
	# 3) Return the two lists

	split_column_values = data[:, split_column] #The column values
	type_of_feature = FEATURE_TYPES[split_column]
	
	if type_of_feature == "continuous":
		data_below = data[split_column_values <= split_value] #Use Boolean Indexing to get the data of interest
		data_above = data[split_column_values > split_value]
	else: #Categorical data
		data_below = data[split_column_values == split_value] #Use Boolean Indexing to get the data of interest
		data_above = data[split_column_values != split_value]

	return data_below, data_above
# --------------------------------------------------------------------------------------------------------------------------



# CALCULATE ENTROPY --------------------------------------------------------------------------------------------------------
# A function that calculates entropy.
def calculate_entropy(data):

	label_column = data[:, -1] #Determine the number of classes
	ignore, counts = np.unique(label_column, return_counts=True) #Determine how many times the classes appear
	probabilities = counts/counts.sum() #An array of the probabilities of each class, numpy element-wise operation "sum" is used

	entropy = sum(probabilities * -np.log2(probabilities)) #The entropy equation, numpy element-wise operation "log2" is used

	return entropy
# --------------------------------------------------------------------------------------------------------------------------



# CALCULATE OVERALL ENTROPY ------------------------------------------------------------------------------------------------
# A function that calculates the overall entropy of the system.
def calculate_overall_entropy(data_below, data_above):

	n_data_points = len(data_below) + len(data_above)

	p_data_below = len(data_below)/n_data_points
	p_data_above = len(data_above)/n_data_points

	overall_entropy = (p_data_below*calculate_entropy(data_below)) + (p_data_above*calculate_entropy(data_above))

	return overall_entropy
# --------------------------------------------------------------------------------------------------------------------------



# DETERMINE BEST SPLIT -----------------------------------------------------------------------------------------------------
# This function determines the best potential value to split the data by. It then returns the best column (datatype) to split by, and
# the best value within that column to split by.
def determine_best_split(data, potential_splits):

	overall_entropy = 999 #Overall entropy set arbitrarily high in order to ensure subsequent code functions properly

	for column_index in potential_splits: #Iterate through the data at the potential splits
		for value in potential_splits[column_index]:
			data_below, data_above = split_data(data, column_index, value)
			current_overall_entropy = calculate_overall_entropy(data_below, data_above)

			if current_overall_entropy <= overall_entropy: #If the split column/value pair under analysis creates the least entropy so far
				overall_entropy = current_overall_entropy
				best_split_column = column_index #Make them the best column/value for splitting
				best_split_value = value

	return best_split_column, best_split_value
# --------------------------------------------------------------------------------------------------------------------------



# DETERMINE FEATURE TYPE ---------------------------------------------------------------------------------------------------
# This function determines if the data is categorical or continuous. It will do this by analyzing the number of datapoints per feature.
# Features with many datapoints (more than some threshold) will be identified as continuous and not categorical.
# Also, features with values that are strings, instead of numbers, will be considered categorical data.
# The feature types will be returned.
def determine_type_of_feature(df):
	feature_types = [] 
	n_unique_values_threshold = 15 #If some feature has more than this threshold of datapoints, we will assume it is a continuous feature
		
	for column in df.columns: #Loop through each column, each feature
		unique_values = df[column].unique() #The number of unique values in the current column
		example_value = unique_values[0] #A sample datapoint from the current column

		if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_threshold):
			feature_types.append("categorical")
		else:
			feature_types.append("continuous")

	return feature_types
# --------------------------------------------------------------------------------------------------------------------------



# DECISION TREE ALGORITHM --------------------------------------------------------------------------------------------------
# The data structure of the decision tree itself is a dictionary where the key is a question and the value is a list of two items.
# Item one is the Yes answer, and item two the No answer to the key-question. In the case of Yes, we have simply the classification,
# in the case of no, we can have a classification in the terminal case, but in most cases we have another dictionary with another question.
# The dictionary keys/questions are the nodes/leaves of the tree, and the lists hold the two paths/branches for each.
# Thus, the tree is made up of sub-trees of form "sub_tree = {question: [yes_answer, no_answer]}" - This will be a recursive function!
# random_subspace added for the Random Forest Algorithm project, not part of this program/project - ignore it if focusing only Decision Trees.
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5, random_subspace=None):

	#data preparations
	if counter == 0: #In the first call of the function
		global COLUMN_HEADERS, FEATURE_TYPES
		COLUMN_HEADERS = df.columns
		FEATURE_TYPES = determine_type_of_feature(df)
		data = df.values #Ensure the data is a numpy 2D array
	else: #Not in the first call of the function
		data = df

	#terminal case
	if check_purity(data) or (len(data) < min_samples) or (counter == max_depth):
		classification = classify_data(data)
		return classification
	#recursive case
	else:
		counter += 1

		#generate data
		potential_splits = get_potential_splits(data, random_subspace)
		split_column, split_value = determine_best_split(data, potential_splits)
		data_below, data_above = split_data(data, split_column, split_value)

		#check for empty data
		if len(data_below) == 0 or len(data_above) == 0:
			classification = classify_data(data)
			return classification

		#instantiate sub-tree
		feature_name = COLUMN_HEADERS[split_column]
		type_of_feature = FEATURE_TYPES[split_column]
		if type_of_feature == "continuous":
			question = "{} <= {}".format(feature_name, split_value) #Create continuous question string
		else:
			question = "{} = {}".format(feature_name, split_value) #Create categorical question string
		sub_tree = {question: []}

		yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth, random_subspace) #Calc pure classification, "data_below"
		no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth, random_subspace) #Calc impure classification, "data_above"
		if yes_answer == no_answer: #If we've reached the terminal case in the "data_above"
			sub_tree = yes_answer #Just choose one of the answers to be the string returned
		else:
			sub_tree = {question: [yes_answer, no_answer]}

		return sub_tree
# --------------------------------------------------------------------------------------------------------------------------



# CLASSIFIER ---------------------------------------------------------------------------------------------------------------
# This function classifies a datapoint based on its features, using the decision tree.
# It does so by taking the example through the tree, asking the question at each node, following resulting paths, and eventually 
# reaching a terminal classification.
# In practical terms this means the following: if our example's feature when put into the decision tree dictionary key formula results in True,
# then we progress to the first element in the value list, otherwise we progress to the second element. If we reach a class we stop. We know we
# have reached a class when we are no longer looking at another dictionary, but a string, float, int, etc.
def classify_example(example, tree):

	#Ask the question
	question = list(tree.keys())[0] #Capture the tree/dictionary's question/key
	feature_name, comparison_operator, value = question.split() #Capture question/key data

	#Ask the node/leaf question
	if comparison_operator =="<=": #If continuous feature
		if example[feature_name] <= float(value): #Always will be less-than (data BELOW) for the left branch or True branch
			answer = tree[question][0]
		else:
			answer = tree[question][1] #False branch
	else: #If categorical feature
		if example[feature_name] == value: #Always will be equal-to for the left branch 
			answer = tree[question][0]
		else:
			answer = tree[question][1] #False branch

	#At this point we have the answer, the next node we are at - be it a classification or a dictionary (subtree)
	if not isinstance(answer, dict): #If answer is NOT a dictionary
		return answer #Then it is a value and so we have reached a terminal case
	else: #Otherwise we are at a sub-tree and shall continue down the tree recursively by repeating the process
		residual_tree = answer
		return classify_example(example, residual_tree)
# --------------------------------------------------------------------------------------------------------------------------



# ACCURACY -----------------------------------------------------------------------------------------------------------------
# This function should calculate the accuracy of the classifier passed to it by using data also passed to it.
def calculate_accuracy(df, tree):

	# 1) Break apart the df into two arrays, one of the features and another of the labels, still 1:1
	# 2) Run the features through the classifier, then compare the output to the labels, if it is right, iterate some "right" variable,
	# 	else iterate some "wrong" variable
	# 3) Calculate % right and % wrong after all items in the list have been iterated

	df["classification"] = df.apply(classify_example, axis=1, args=(tree,)) #Create a column in the df containing the classification for each example
	df["classification_correct"] = (df.classification == df.label) #Create a column of Booleans comparing classification w/ corresponding labels

	accuracy = df.classification_correct.mean() #The mean of the classification-label comparisons is the accuracy

	return accuracy
# --------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
#-------------------------------------Random Forest Algorithm Helper Functions----------------------------------------------

def predict_example(example, tree):

	#Ask the question
	question = list(tree.keys())[0] #Capture the tree/dictionary's question/key
	feature_name, comparison_operator, value = question.split() #Capture question/key data

	#Ask the node/leaf question
	if comparison_operator =="<=": #If continuous feature
		if example[feature_name] <= float(value): #Always will be less-than (data BELOW) for the left branch or True branch
			answer = tree[question][0]
		else:
			answer = tree[question][1] #False branch
	else: #If categorical feature
		if example[feature_name] == value: #Always will be equal-to for the left branch 
			answer = tree[question][0]
		else:
			answer = tree[question][1] #False branch

	#At this point we have the answer, the next node we are at - be it a classification or a dictionary (subtree)
	if not isinstance(answer, dict): #If answer is NOT a dictionary
		return answer #Then it is a value and so we have reached a terminal case
	else: #Otherwise we are at a sub-tree and shall continue down the tree recursively by repeating the process
		residual_tree = answer
		return predict_example(example, residual_tree)

# --------------------------------------------------------------------------------------------------------------------------

def decision_tree_predictions(test_df, tree):
	predictions = test_df.apply(predict_example, args=(tree,), axis=1)
	return predictions

# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------



if __name__ == '__main__':

	# LOAD AND PREPARE IRIS DATA SET -------------------------------------------------------------------------------------------
	random.seed(0)
	df = pd.read_csv("Iris.csv")
	df = df.drop("Id", axis=1)
	df = df.rename(columns={"species":"label"})
	print(df.head())
	# --------------------------------------------------------------------------------------------------------------------------

	# # LOAD AND PREPARE TITANIC DATA SET ----------------------------------------------------------------------------------------
	# random.seed(0)
	# df = pd.read_csv("Titanic.csv")
	# df["label"] = df.Survived
	# df = df.drop(["PassengerID", "Survived", "Name", "Ticket", "Cabin"], axis=1)
	# median_age = df.Age.median()
	# mode_embarked = df.Embarked.mode()[0]
	# df = df.fillna({"Age": median_age, "Embarked": mode_embarked})
	# # --------------------------------------------------------------------------------------------------------------------------

	# MAIN RUN -----------------------------------------------------------------------------------------------------------------
	train_df, test_df = train_test_split(df, test_size=0.2)
	tree = decision_tree_algorithm(train_df)
	accuracy = calculate_accuracy(test_df, tree)
	print(test_df.head())
	print(df.head(),"\n")
	pprint(tree, width=50)
	print("\n",accuracy)
	# --------------------------------------------------------------------------------------------------------------------------