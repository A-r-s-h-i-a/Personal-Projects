# Melody Generation
# Arshia Firouzi
# 10/06/2021
#
# Training the LSTM Network

import numpy as np
import tensorflow.keras as keras
from PreProcess import get_vocabulary_size, generate_training_sequences, SEQUENCE_LENGTH, TRAINING_TREBLE_FILE, TRAINING_BASS_FILE

MODEL_SAVE_PATH = "NDS_MelodyGenerator.h5"
LSTM_UNITS = 256 #Number of neurons
DENSE_UNITS = 128
LOSS_FN = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 40
BATCH_SIZE = 64



# Gathers the training sequences, builds the LSTM network, trains the network, then saves it
def train(LSTM_neurons=LSTM_UNITS, Dense_neurons=DENSE_UNITS, loss_fn=LOSS_FN, learning_rate=LEARNING_RATE):

	# Generate the training sequences
	print("\nImporting training data...")
	vocabulary_size = get_vocabulary_size()
	treble_inputs, treble_targets = generate_training_sequences(TRAINING_TREBLE_FILE, SEQUENCE_LENGTH, vocabulary_size)
	bass_inputs, bass_targets = generate_training_sequences(TRAINING_BASS_FILE, SEQUENCE_LENGTH, vocabulary_size)
	# Define the inputs and targets appropriately
	inputs = [treble_inputs, bass_inputs]
	targets = [treble_targets, bass_targets]
	print("\n\n\n", np.shape(treble_inputs), np.shape(treble_targets), "\n", np.shape(bass_inputs), np.shape(bass_targets), "\n\n\n")
	print("Training data imported!")

	# Build the LSTM network
	print("\nModel training has begun...")
	model = build_model(vocabulary_size, LSTM_neurons, Dense_neurons, loss_fn, learning_rate)
	print("Model built, summary below:	\n")
	model.summary()
	print("\n")

	# Train the model
	model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

	# Save the model
	model.save(MODEL_SAVE_PATH)
	print("Model trained and saved!\n")



# Builds the LSTM network
def build_model(vocabulary_size, LSTM_neurons, Dense_neurons, loss_fn, learning_rate):

	# Create model architecture, 2 inputs pathways which eventually join - one for treble and one for bass data
	inpt1 = keras.layers.Input(shape=(None, vocabulary_size)) #vocabulary size is the number of categories/notes
	inpt2 = keras.layers.Input(shape=(None, vocabulary_size))
	# Pass through LSTM layers, with dropout to avoid overfitting
	L1 = keras.layers.LSTM(LSTM_neurons)(inpt1)
	L2 = keras.layers.LSTM(LSTM_neurons)(inpt2)
	do1 = keras.layers.Dropout(0.2)(L1) #Dropout is a technique to avoid overfitting, randomly sets input units to 0 when training
	do2 = keras.layers.Dropout(0.2)(L2)
	# Combine the outputs of the LSTMs
	comb = keras.layers.Concatenate()([do1, do2])
	# Pass through a couple dense layers, another dropout, then output
	D0 = keras.layers.Dense(Dense_neurons, activation="relu")(comb)
	do3 = keras.layers.Dropout(0.2)(D0)
	D1 = keras.layers.Dense(Dense_neurons, activation="relu")(do3)
	D2 = keras.layers.Dense(Dense_neurons, activation="relu")(D1)
	outpt1 = keras.layers.Dense(vocabulary_size, activation="softmax")(D2)
	outpt2 = keras.layers.Dense(vocabulary_size, activation="softmax")(D2)

	model = keras.Model(inputs=[inpt1, inpt2], outputs=[outpt1, outpt2])

	# Compile the model
	model.compile(loss=loss_fn,
				optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
				metrics=["accuracy"])

	return model



if __name__ == "__main__":
	train()
