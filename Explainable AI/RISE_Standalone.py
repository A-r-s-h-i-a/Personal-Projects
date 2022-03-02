# Arshia Firouzi
# 08/13/2021
# RISE - Randomized Input Sampling for Explanation of Black-box Models
# https://arxiv.org/pdf/1806.07421.pdf

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow.keras import Model #For creating TF-supported classes
from tensorflow.keras.layers import Input, Dense, Flatten, ReLU, Conv2D, MaxPool2D, BatchNormalization #Necessary FC/Conv NN functions



class RISE(object):
	"""
	Randomized Input Sampling for Explanation [RISE]
		RISE is a technique for identifying what the key decision-making factors are for a Computer Vision model
	in arriving at its output for some image. This is especially difficult as CNNs, especially larger ones, 
	operate essentially as black-boxes. One way to measure the importance of a pixel/collection of pixels to the 
	decision making process of a Computer Vision model is to perturb it and observe how it affects its output. 
	RISE uses a similar approach. 
		The algorithm first randomly generates many boolean masks, then multiplies them by the input image being
	analyzed. This results in many respectively masked versions of the image, where random combinations of the
	pixels have been turned OFF. Next, in a loop, the altered, masked images are passed through the Computer
	Vision model, and the output is captured. This output value is then assigned in a weighted fashion as the
	Importance of each ON pixel for the masked image in that loop's iteration, and the loop repeats. These pixel
	Importance values are captured additively in a 2D matrix object (with the same dimensions as the image) called a 
	Saliency Map that is the final output of RISE. The areas where the Saliency Map has the greatest value (and
	thus the summed Importance values are greatest) are were, theoretically, the greatest decision-making factors lie.

	__Attributes__
	n_masks				::	The number of masks to be generated
	m_shape 			::	The shape of the mask to be used, logically it must match that of the image being processed
	masks 				::	A matrix of the randomly generated masks
	masked 				::	A matrix of the respective versions of the image masked by the randomly generated masks
	rm_train			::	How many datapoints from the training dataset to remove, enables faster program execution
	rm_test				::	How many datapoints from the testing dataset to remove, enables faster program execution
	ds_train 			::	Training dataset
	ds_test 			::	Test dataset

	__Methods__
	create_mask			::	Creates a single mask by randomly generating a small one, then expanding it to one of size
								HxW via bilinear interpolation
	generate_masks		::	Utilizes create_mask to generate 'n_masks' number of masks, storing them in the 'masks' attribute
	mask_img 			:: 	Process the image with each mask, creating the self.masked attribute
	load_data			::	Loads and parses the provided dataset (tf's cycle_gan, apples & oranges) into the respective network attributes
	get_datapoint		::	Returns the indexed datapoint from the downloaded dataset
	get_weighted_sum	::	Iteratively calculates the Importance of each pixel from the masked image and the model's
								prediction for it, combines the Importance Maps into the final Saliency Map
	"""
	def __init__(self, n_masks, m_shape):
		super(RISE, self).__init__()
		self.n_masks = n_masks
		self.m_shape = m_shape #Expected WxH format

	def create_mask(self, H, W, h=7, w=7):
		#Probability of Bool False/0 in the mask (obscure)
		p=0.5
		#Create random, small, hxw mask
		mask = np.random.choice(a=[0,1], size=(h,w), p=[p, 1-p])

		#Resize the mask to HxW using Bilinear Interpolation
		resample = Image.BILINEAR
		mask = Image.fromarray(mask*255) #expand to 255 vals for bilinear interpolation to be more continuous
		mask = mask.resize((H+h, W+w), resample=resample)
		mask = np.array(mask)

		#Crop down columns by h and rows by w randomly
		h_crop = np.random.randint(0, h+1)
		w_crop = np.random.randint(0, w+1)
		mask = mask[h_crop:H+h_crop, w_crop:W+w_crop]

		#Normalize back to 0/1 values
		mask = mask/np.max(mask)

		return mask

	def generate_masks(self):
		self.masks = np.array([self.create_mask(H=224,W=224)])
		for i in tqdm(range(self.n_masks-1), desc="Generating Masks"):
			mask = np.array([self.create_mask(H=224,W=224)])
			self.masks = np.vstack((self.masks, mask))

	def mask_img(self, img):
		self.masked = np.expand_dims(np.empty(img.shape), axis=0)
		for mask in tqdm(self.masks, desc="Masking Image"):
			#Multiply mask through each color channel
			img_masked = img.copy()
			img_masked[:,:,0] = img_masked[:,:,0] * mask
			img_masked[:,:,1] = img_masked[:,:,1] * mask
			img_masked[:,:,2] = img_masked[:,:,2] * mask
			img_masked = np.array([img_masked])
			#Add masked image to masked images
			self.masked = np.vstack((self.masked, img_masked))

		#Remove empty first item of self.masked
		self.masked = np.delete(self.masked, obj=0, axis=0)

		return self.masked

	def get_weighted_sum(self, pred, class_index):
		wsum = np.zeros(self.masks[0].shape)

		#Calculate sum of masks
		for i, mask in enumerate(self.masks):
			vals = mask * pred[i, class_index]
			wsum += vals

		return wsum

	def load_data(self, batch_size, rm_train=0, rm_test=0):
		#NOTE: Shuffling is not performed so repeatable indexing is possible
		#Get the Apples & Oranges Datasets
		(ds_trainA, ds_testA, ds_trainB, ds_testB), ds_info = tfds.load(
												"cycle_gan",
												split=['trainA', 'testA', 'trainB', 'testB'], #Defines ds_train, ds_test order
												shuffle_files=False, #Set to True, files are preordered by tf in a certain order for optimization
												with_info=True, #Tells the tfds.load to return info (ds_info)
												as_supervised=True, #Returns tuples containing labels (img, label) for train and test, must be false
																	#	to use the "show_examples" method
												)

		#Reduce training dataset by number of items rm_train, and test by rm_test
		ds_trainA = ds_trainA.skip(rm_train)
		ds_trainB = ds_trainB.skip(rm_train)
		ds_testA = ds_testA.skip(rm_test)
		ds_testB = ds_testB.skip(rm_test)

		#Combine and Shuffle the Datasets, set reshuffle to FALSE so that ordering stays the same for RISE
		self.ds_train = ds_trainA.concatenate(ds_trainB)
		training_buffer = ds_info.splits['trainA'].num_examples + ds_info.splits['trainB'].num_examples - (2*rm_train)
		self.ds_train = self.ds_train.batch(batch_size=batch_size, drop_remainder=True)
		self.ds_train = self.ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)
		self.ds_test = ds_testA.concatenate(ds_testB)
		testing_buffer = ds_info.splits['testA'].num_examples + ds_info.splits['testB'].num_examples - (2*rm_test)
		self.ds_test = self.ds_test.batch(batch_size=batch_size, drop_remainder=True)
		self.ds_test = self.ds_test.prefetch(buffer_size=tf.data.AUTOTUNE)

	def get_datapoint(self, idx):
		datapoint = list(self.ds_train.unbatch().as_numpy_iterator())[idx][0]
		# print("DATAPOINT: ", datapoint, datapoint.shape)
		return datapoint



if __name__ == '__main__':

	print("\n\n\n")
	
	#Load the pre-trained VGG16 CV Model
	model = tf.keras.applications.vgg16.VGG16()
	print("VGG16 Model Loaded")

	#Initialize RISE Object
	XAI = RISE(n_masks=4000, m_shape=(224,224)) #VGG16 accepts 224x224 inputs, thus we must work with those dimensions
	print("RISE Object Instantiated")

	#Generate RISE Masks
	XAI.generate_masks()
	print("Masks Generated")

	#Load dataset
	XAI.load_data(batch_size=64)

	#Get the image to be XAI analyzed
	datapoint = XAI.get_datapoint(idx=121)
	#Fit datapoint to input dimensions (224x224) of the VGG16 model
	datapoint = datapoint[0:224,0:224,:]
	print("Datapoint Loaded")

	#Mask the image
	masked_img = XAI.mask_img(datapoint)

	#Run the unmasked and masked versions of the image through the model
	pred_unmasked = model.predict(np.expand_dims(datapoint, axis=0))[0]
	pred_masked = model.predict(masked_img, verbose=1)

	#Get the top prediction name/index from the model (equals the largest output, VGG16 output layer is softmax)
	pred_decoded = tf.keras.applications.vgg16.decode_predictions(pred_unmasked[np.newaxis, :], top=1)
	class_name = pred_decoded[0][0][1]
	class_index = np.argmax(pred_unmasked)
	print("Class Name:", class_name)
	print("Class Index: ", class_index)

	#Analyze the processed masked image with RISE
	Saliency_Map = XAI.get_weighted_sum(pred=pred_masked, class_index=class_index)

	#Plot the resulting saliency map over the original image
	plt.imshow(datapoint)
	plt.imshow(Saliency_Map, cmap=plt.cm.jet, alpha=0.5)
	plt.show()