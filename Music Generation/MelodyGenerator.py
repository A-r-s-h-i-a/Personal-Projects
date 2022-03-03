# Melody Generation
# Arshia Firouzi
# 11/03/2021
#
# Utilities for generating melodies from the trained LSTM model

import json
import numpy as np
import music21 as m21
import tensorflow.keras as keras
from Training import MODEL_SAVE_PATH
from PreProcess import SEQUENCE_LENGTH, MAPPING_FILE



def is_list(item):
	try:
		list(item)
		return True
	except ValueError:
		return False



class MelodyGenerator:

	def __init__(self, model_path):

		self.model_path = model_path
		self.model = keras.models.load_model(model_path)

		with open(MAPPING_FILE, "r") as fp:
			self._mappings = json.load(fp)

		self._start_symbols = ["/"] * SEQUENCE_LENGTH

	def generate_melody(self, seed_treble, seed_bass, num_steps, max_sequence_length, temperature):

		# Create the seed (with start symbols)
		seed_treble = seed_treble.split()
		seed_bass = seed_bass.split()
		# Check that the seeds are of equal time length
		if len(seed_treble) != len(seed_bass):
			print("\n\n*****ERROR! Seeds do not span the same length of time! ERROR!*****\n\n")
			return
		# Continue creating the seed...
		seed_treble = self._start_symbols + seed_treble
		seed_bass = self._start_symbols + seed_bass
		# Initialize the melody to the seed
		melody = [seed_treble, seed_bass]

		# Map the seed values to integers
		seed_treble = [self._mappings[symbol] for symbol in seed_treble]
		seed_bass = [self._mappings[symbol] for symbol in seed_bass]

		# Generation loop
		for _ in range(num_steps):

			# Limit the seed to only the last max_sequence_length items
			seed_treble = seed_treble[-max_sequence_length:]
			seed_bass = seed_bass[-max_sequence_length:]

			# One-hot encode the values
			onehot_seed_treble = keras.utils.to_categorical(seed_treble, num_classes=len(self._mappings))
			onehot_seed_treble = onehot_seed_treble[np.newaxis, ...] #Add an extra batch dimension for keras
			onehot_seed_bass = keras.utils.to_categorical(seed_bass, num_classes=len(self._mappings))
			onehot_seed_bass = onehot_seed_bass[np.newaxis, ...]

			# Make a prediction
			inpt = [onehot_seed_treble, onehot_seed_bass]
			probabilities = self.model.predict(inpt)
			probabilities_treble = probabilities[0][0]
			probabilities_bass = probabilities[1][0]
			output_int_treble = self._sample_with_temperature(probabilities_treble, temperature) #Model has a Softmax output
			output_int_bass = self._sample_with_temperature(probabilities_bass, temperature)

			# Update the seed
			seed_treble.append(output_int_treble)
			seed_bass.append(output_int_bass)

			# Map the prediction-output to a symbol
			output_symbol_treble = [k for k, v in self._mappings.items() if v==output_int_treble][0]
			output_symbol_bass = [k for k, v in self._mappings.items() if v==output_int_bass][0]

			# Check if we are at the end of the melody, stop if so!
			if output_symbol_treble == "/" or output_symbol_bass == "/":
				break

			# If we are NOT at the end of the melody, update the melody with the new note
			melody[0].append(output_symbol_treble)
			melody[1].append(output_symbol_bass)

		return melody

	# This function samples an index from "probabilities" (a softmax list) using the "temperature" value. As the 
	#	value of temperature goes to infinity, then this function acts to randomly choose an index. As it goes to
	#	zero, the function acts as an argmax(probabilities). Lastly, when temperature equals 1, then the values at 
	#	each index of probabilities are the probabilities that each will be chosen respectively. This function
	#	enables sampling with a tunable variable for exploration/unpredictability.
	def _sample_with_temperature(self, probabilities, temperature):
		
		predictions = np.log(probabilities) / temperature
		probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

		choices = range(len(probabilities))
		index = np.random.choice(choices, p=probabilities)

		return index

	# Saves the melody in a given (non symbolic) format
	# 	"step_duration" is the time series encoding "time_step"
	def save_melody(self, melody, file_name="melody.mid", step_duration=0.25, format="midi"):

		# Create the music21 stream and treble/bass clefs
		score = m21.stream.Score(id="Main") #Default is 4/4 time signature, Cmaj
		treble_clef = m21.stream.Part(id="Treble")
		bass_clef = m21.stream.Part(id="Bass")

		# Parse all the symbols in the melody and create note/rest objects
		start_symbol_treble = None
		start_symbol_bass = None
		treble_step_counter = 1
		bass_step_counter = 1
		for idx in range(len(melody[0])):
			# Get the treble and bass notes
			treble_note = melody[0][idx]
			bass_note = melody[1][idx]

			# Handle case where note == /
			if treble_note == "/" or bass_note == "/":
				continue

			# Handle case where treble note == note/rest
			if treble_note != "_" or (idx+1)==len(melody[0]):

				# Ensure we're not dealing with the very first note
				if start_symbol_treble is not None:
					quarter_length_duration = step_duration * treble_step_counter

					# Handle Rest
					if start_symbol_treble == "r":
						treble_m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

					# Handle Chord
					elif is_list(start_symbol_treble) and len(list(start_symbol_treble))>3: #Notes are strings and thus lists! Must be a long enough list
						chord = start_symbol_treble[1:-1].split(",") #Split the string into a list appropriately
						chord = [int(item) for item in chord] #Convert strings in the list into integers (the MIDI values) 
						treble_m21_event = m21.chord.Chord(chord, quarterLength=quarter_length_duration)

					# Handle Note
					else:
						treble_m21_event = m21.note.Note(int(start_symbol_treble), quarterLength=quarter_length_duration)

					# Append note/rest to stream
					treble_clef.append([treble_m21_event])

					# Reset the step counter
					treble_step_counter = 1

				# Update start symbol
				start_symbol_treble = treble_note

			# Handle case where note == "_" (prolongation note)
			else:
				treble_step_counter += 1

			# Repeat everything for the bass note, handle case where the note == a note/rest
			if bass_note != "_" or (idx+1)==len(melody[0]):
				
				# Check/ensure we're not dealing with the very first note
				if start_symbol_bass is not None:
					quarter_length_duration = step_duration * bass_step_counter

					# Handle Rest
					if start_symbol_bass == "r":
						bass_m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

					# Handle Chord
					elif is_list(start_symbol_bass) and len(list(start_symbol_bass))>3: #Notes are strings and thus lists! Must be a long enough list
						chord = start_symbol_bass[1:-1].split(",") #Split the string into a list appropriately
						chord = [int(item) for item in chord] #Convert strings in the list into integers (the MIDI values) 
						bass_m21_event = m21.chord.Chord(chord, quarterLength=quarter_length_duration)

					# Handle Note
					else:
						bass_m21_event = m21.note.Note(int(start_symbol_bass), quarterLength=quarter_length_duration)

					# Append note/rest to stream
					bass_clef.append([bass_m21_event])

					# Reset the step counter
					bass_step_counter = 1

				# Update start symbol
				start_symbol_bass = bass_note

			# Handle case where note == "_" (prolongation note)
			else:
				bass_step_counter += 1

		# Write the m21 stream to a file with the input format
		score.insert(0, treble_clef)
		score.insert(0, bass_clef)
		score.write(format, file_name)



if __name__ == "__main__":
	mg = MelodyGenerator(model_path=MODEL_SAVE_PATH)
	seed_treble = "74 _ _ _ 71 _ _ _ 79 _ _ _ 76 _ _ _ 81 _ _ _ _ _ _ _ 81 _ _ _ _ _ _ _"
	seed_bass = "[50,53,57] _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ [45,48,52] _ _ _ _ _ _ _ _ _ _ _ _ _ _ _"
	melody = mg.generate_melody(seed_treble=seed_treble, seed_bass=seed_bass, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, temperature=1.5)
	print(melody[0], len(melody[0]), "\n\n", melody[1], len(melody[1]), "\n\n\n")
	mg.save_melody(melody, file_name="test_temp15.mid")