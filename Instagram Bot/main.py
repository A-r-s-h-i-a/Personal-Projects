# Main run-file for the instagram bot
# Posts a generated image every ~60 minutes

import os
import time
import torch
import random
import torchvision
import torchvision.transforms as T

from bigBadBot import igBot
from ProGAN import generate_image
from torchvision.utils import save_image
from utils import USE_GPU, SAVE_IMG_PATH, NUM_IMAGES, MODEL, IMG_CAPTIONS



if __name__ == "__main__":

	# # Generate images for posting to Instagram
	# for i in range(NUM_IMAGES):
	# 	# Generate and resize an image each loop
	# 	transform = T.Resize((256, 256))
	# 	img = generate_image()
	# 	resized_img = transform(img)

	# 	# Save the generated image as a jpeg
	# 	SAVE_IMG_PATH_i = SAVE_IMG_PATH + "/" + str(i) + ".jpeg"
	# 	save_image(resized_img.clamp(min=-1, max=1),
	# 		SAVE_IMG_PATH_i, nrow=NUM_IMAGES, scale_each=True,
	# 		normalize=True)

	# Launch Chrome and login to Instagram
	bot = igBot()
	bot.launch_chrome()
	print("\nChrome Launched\n")
	bot.login_ig()
	print("\nLogged In to Instagram\n")

	# Loop every ~60 minutes
	for i in range(NUM_IMAGES):
		# Attempt a post
		caption_idx = min(i, NUM_IMAGES-1)
		print("Attempting to upload image number " + str(i) + ".")
		try:
			bot.upload_photo(n=i, caption_text=IMG_CAPTIONS[caption_idx])
			print("\nUploaded Succeeded!\n")
			time.sleep(random.randrange(60*50, 60*70))
		except Exception as err:
			print("\nUpload Failed!\n")
			print(err)
			print("\n\n\n")
		# Return Chrome to home screen
		bot.return_ig_home()
		# Sleep for 50-70 minutes

	# Before ending the program, kill the app-controlled Chrome
	bot.kill_chrome()
	print("\nKilled Chrome\n")