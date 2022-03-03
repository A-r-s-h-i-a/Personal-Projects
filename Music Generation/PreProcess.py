# Melody Generation
# Arshia Firouzi
# 10/12/2021
#
# Data Preprocessing
# The dataset consists of miscellaneous piano pieces
# The files are downloadable here: https://github.com/jukedeck/nottingham-dataset

import os
import json
import numpy as np
import music21 as m21
import tensorflow.keras as keras



NDS_DATASET_PATH = "Pieces/melody_and_chords"
PREPROCESSED_TREBLE_PATH = "Pieces/preprocessed/treble"
PREPROCESSED_BASS_PATH = "Pieces/preprocessed/bass"
TRAINING_TREBLE_FILE = "Treble_Training_Data"
TRAINING_BASS_FILE = "Bass_Training_Data"
ACCEPTABLE_NOTE_DURATIONS = [i*0.25+0.25 for i in range(24)] #1 is Quarter Note, ranges from a 16th Note to a Whole Note
MAPPING_FILE = "NDS_mapping.json"
SEQUENCE_LENGTH = 64 #The length of the sequence you want the LSTM to learn



# Loads the songs from the dataset
def load_songs_in_midi(dataset_path):
	
	songs = []
	filenames = []

	# Walk through all the files (songs) in dataset, and load them using music21-
	for path, subdirs, files in os.walk(dataset_path):
		for file in files:
			if file[-3:] == "mid": #If the file is a midi file
				song = m21.converter.parse(os.path.join(path, file)) #Capture the song as a music21 "Score" object
				songs.append(song)
				filenames.append(file)

	return songs, filenames



# Check for if the song's notes are all within the acceptable durations
def check_note_durations(acceptable_note_durations, song):

	# Split the grand staff into the Treble and Bass parts (first and second respectively)
	parts = song.getElementsByClass(m21.stream.Part)

	# Walk through each staff, analyzing the notes for their durations
	for part in parts:
		for event in part.flat.notesAndRests: #Flatten score, filter out notes and rests only
			# If a note is not within the acceptable durations, return false
			if event.duration.quarterLength not in acceptable_note_durations:
				return False

	return True



# Check if the song has two Staffs, a Treble and a Bass
def check_Staffs(song):

	# Capture the first measure
	parts = song.getElementsByClass(m21.stream.Part)
	part1 = parts[0]

	# Examine it to see if there are two staffs, return False if not
	if len(parts)==1:
		return False
	else:
		return True



# Transposes the song to Cmaj or Amin if it is in a major or minor key respectively
def transpose(song):

	# Get the key of the song
	parts = song.getElementsByClass(m21.stream.Part)
	measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
	key = measures_part0[0][2] #The key is at index 4 of the measure

	# Estimate the key if necessary
	if not isinstance(key, m21.key.Key):
		key = song.analyze("key")

	# Calculate the interval for transposition (by measuring distance between the key's tonic and the desired pitch)
	# If the piece is in a major mode then transpose to Cmaj, if it is in a minor mode then to Amin
	if key.mode == "major":
		interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
	elif key.mode == "minor":
		interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

	# Perform the transposition
	transposed_song = song.transpose(interval)

	return transposed_song



# Encode songs into a MIDI time series representation, sampled every time-step ("_" represents a held note, "r" a rest)
#	e.g. A note with pitch 60 and duration 1, with a 0.25 time-step, is represented as [60, "_", "_", "_"]
def encode_song(song, time_step=0.25):

	encoded_treble, encoded_bass, encoded_parts = [], [], []

	# Split the grand staff into the Treble and Bass parts (first and second respectively)
	parts = song.getElementsByClass(m21.stream.Part)

	# For both parts of the song, walk through each note/rest chronologically and encode it
	for part in parts:
		encoded_part = []
		for event in part.flat.notesAndRests: #Flatten score, filter out notes and rests only
			# Handle notes
			if isinstance(event, m21.note.Note):
				symbol = event.pitch.midi

			# Handle chords
			elif isinstance(event, m21.chord.Chord):
				chord_tuple = event.pitches
				symbol = [idx.midi for idx in chord_tuple]
				symbol = str(symbol).replace(" ","") #Convert to string here so that the list can be stripped of whitespace easily

			# Handle rests
			elif isinstance(event, m21.note.Rest):
				symbol = "r"

			# Convert note/rest into our time-series representation
			steps = int(event.duration.quarterLength / time_step)
			for step in range(steps):
				if step == 0: #Initial instance of note
					encoded_part.append(symbol)
				else: #Held note
					encoded_part.append("_")
		encoded_parts.append(encoded_part)

	# Create encoded songs, a list of string items with polyphonic items included
	for i in range(len(encoded_parts[0])):
		# Capture the note (mono or polyphonic)
		item = encoded_parts[0][i]

		# Check if it is a number or string and treat it appropriately
		if isinstance(item, str):
			# If it is already a string, keep it as-is
			string_cast_item = item
		else:
			# If it is a number, convert it to a string
			string_cast_item = str(item)

		# Now append the strings to the appropriate encoded_song vector
		encoded_treble.append(string_cast_item)

	# Repeat for the Bass
	for i in range(len(encoded_parts[1])):
		# Capture the note (mono or polyphonic)
		item = encoded_parts[1][i]

		# Check if it is a number or string and treat it appropriately
		if isinstance(item, str):
			# If it is already a string, keep it as-is
			string_cast_item = item
		else:
			# If it is a number, convert it to a string
			string_cast_item = str(item)

		# Now append the strings to the appropriate encoded_song vector
		encoded_bass.append(string_cast_item)

	# Cast encoded parts to a string, if staffs are inequal in length set the length flag to True
	len_flag = False
	if len(encoded_treble) != len(encoded_bass):
		len_flag = True
	encoded_treble = " ".join(map(str, encoded_treble))
	encoded_bass = " ".join(map(str, encoded_bass))

	return encoded_treble, encoded_bass, len_flag



# Preprocessing function which downloads the songs, then filters them, transposes them to Cmaj or Amin,
#	encodes them into a time-series representation (where notes are sampled every time-step, a "_"
#	character indicates a held note, and an "r" a rest), and finally saves them
def preprocess(unprocessed_dataset_path):

	# Load the songs
	print("\nLoading songs...")
	songs, filenames = load_songs_in_midi(unprocessed_dataset_path)
	print(f"Loaded {len(songs)} songs!")

	# Transpose, encode, then save the songs
	print("\nTransposing and encoding songs...")
	skipped = 0
	for idx, song in enumerate(songs):

		# If the song contains notes with noncompliant durations, skip it
		if check_note_durations(ACCEPTABLE_NOTE_DURATIONS, song) is False:
			skipped+=1
			continue
		
		# If the song does not contain both a Treble and Bass staff, skip it
		if check_Staffs(song) is False:
			skipped+=1
			continue

		# Transpose the song to Cmaj or Amin if it is in a major or minor key respectively
		song = transpose(song)
	
		# Encode the song with a time-series representation
		encoded_treble, encoded_bass, len_flag = encode_song(song)
		# Skip the song if it has unequal time-length parts
		if len_flag == True:
			skipped+=1
			continue

		# Finally, save the encoded/transposed songs to text files for access
		treble_save_path = os.path.join(PREPROCESSED_TREBLE_PATH, "treble_"+str(idx))
		bass_save_path = os.path.join(PREPROCESSED_BASS_PATH, "bass_"+str(idx))
		with open(treble_save_path, "w") as fp:
			fp.write(encoded_treble)
		with open(bass_save_path, "w") as fp:
			fp.write(encoded_bass)

	print(f"{skipped} out of {len(songs)} songs skipped!")
	print("Encoding and transposition complete!")



# Function to load processed song files (their time-series data) as a string
def load(file_path):
	
	# Read file at given location
	with open(file_path, "r") as fp:
		song = fp.read()

	return song



# Compile the dataset of encoded songs into a single string then file, with each song separated by delimiters
def create_single_file_dataset(processed_dataset_path, output_file_path, sequence_length):

	new_song_delimiter = "/ " * sequence_length #We want the delimiter to be the size of the sequence length
	songs = "" #String which will contain all the dataset data

	# Walk through each file in the processed dataset, load the songs to a single string with intervening delimiters
	for path, _, files in os.walk(processed_dataset_path):
		for file in files:
			file_path = os.path.join(path, file)
			song = load(file_path)
			songs = songs + song + " " + new_song_delimiter

	# Edit out the empty space at end of songs string
	songs = songs[:-1]

	# Save the songs string to a file
	with open(output_file_path, "w") as fp:
		fp.write(songs)

	return songs



# Create a mapping between encoded time-series values, and ints (as ints are digestible by a net)
def create_mapping(songs_treble, songs_bass):

	mappings = {}

	# Identify the vocabulary of the time-series
	songs_treble = songs_treble.split()
	songs_bass = songs_bass.split()
	songs_joined = songs_treble + songs_bass #Combine treble and bass to have a single vector to work off of
	vocabulary = list(set(songs_joined)) #Reduce instances of items in the list to only one, no repititions, then convert to list

	# Create the mappings of Vocabulary:Integers
	for i, symbol in enumerate(vocabulary):
		mappings[symbol] = i

	# Save the mapping to a json file for easy reference later
	with open(MAPPING_FILE, "w") as fp:
		json.dump(mappings, fp, indent=4)



# A function to get the number of unique notes found in the dataset
def get_vocabulary_size():
	# Load the mapping json file, which should only include unique notes in a dictionary fashion
	note_mapping_file = load(MAPPING_FILE)
	
	# The file should have loaded as a string, split it by its dictionary-like colon character
	temp_file = note_mapping_file.split(":")

	# There should be only one extra item than the length of this list
	vocabulary_size = len(temp_file) - 1

	return vocabulary_size



# Use the mapping of symbols:ints to convert the songs file to a data representation with only ints
def convert_songs_to_int(songs):

	int_songs = []

	# Load mappings for reference
	with open(MAPPING_FILE, "r") as fp:
		mappings = json.load(fp)

	# Cast songs string to a list
	songs = songs.split()

	# Map each value in the songs list to an int using the mappings
	for symbol in songs:
		int_songs.append(mappings[symbol])

	return int_songs



# Generate fixed-length, labeled data sequences for LSTM training - the label is the target next-value/note
def generate_training_sequences(target_dataset_path, sequence_length, vocabulary_size):

	# Load songs and convert them to an integer representation
	songs = load(target_dataset_path)
	int_songs = convert_songs_to_int(songs)

	# Generate the training sequences
	inputs = []
	targets = []
	num_sequences = len(int_songs) - sequence_length
	# Slide through the song int list with a fixed length (of sequence length) setting the current range as
	#	the input, and the following value as the associated target
	for i in range(num_sequences):
		inputs.append(int_songs[i:i+sequence_length])
		targets.append(int_songs[i+sequence_length])

	# One-hot encode the sequences
	inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size) #One-Hot encodes
	targets = np.array(targets)

	# Inputs now has shape (# of sequences, sequence length, vocabulary size)
	return inputs, targets



def main():
	preprocess(NDS_DATASET_PATH)
	print("\nBuilding training data...")
	songs_treble = create_single_file_dataset(PREPROCESSED_TREBLE_PATH, TRAINING_TREBLE_FILE, SEQUENCE_LENGTH)
	songs_bass = create_single_file_dataset(PREPROCESSED_BASS_PATH, TRAINING_BASS_FILE, SEQUENCE_LENGTH)
	create_mapping(songs_treble, songs_bass) #Get the number of unique values (notes) possible
	vocabulary_size = get_vocabulary_size()
	print(f"{vocabulary_size} unique notes/chords identified in the dataset.")
	treble_inputs, treble_targets = generate_training_sequences(TRAINING_TREBLE_FILE, SEQUENCE_LENGTH, vocabulary_size)
	bass_inputs, bass_targets = generate_training_sequences(TRAINING_BASS_FILE, SEQUENCE_LENGTH, vocabulary_size)
	print("Training data built!\n")



if __name__ == "__main__":
	main()