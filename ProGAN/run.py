import torch
import config
import utils
import numpy as np
from math import log2
from model import Generator
import torch.optim as optim
import matplotlib.pyplot as plt



if __name__ == "__main__":
	# Setup parameters
	alpha = 1 #Set alpha to 1 so that we only use the last layer
	num_steps = int(log2(config.IMAGE_SIZE/4)) #The number of steps through the Generator network
	x = torch.randn((1, config.Z_DIM, 1, 1)) #Random Generator Input
	gen_weights_path = "DLd_Weights/generator.pth" #Generator weights path
	# gen_weights_path = "My_Results/generator.pth" #Generator weights path


	# Get Generator output
	gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG) #Initialize Generator
	opt_gen = optim.Adam(gen.parameters(), lr=0.001, betas=(0.0, 0.99)) #N/A, just necessary to load weights
	utils.load_checkpoint(gen_weights_path, gen, opt_gen, 0.001) #Load weights
	out = gen(x, alpha=alpha, steps=num_steps)

	# Transform output
	out = out[0].permute(1,2,0) #Rearrange columns/rows
	out = out.detach().numpy() #Convert to numpy
	out += np.abs(np.amin(out)) #Scale Up
	out = out / np.amax(out) #Scale Down

	plt.imshow(out)
	plt.show()