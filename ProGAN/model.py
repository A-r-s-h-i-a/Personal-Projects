import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2



# The Disc/Gen layer output shapes, largest value (512) scaled to 1, taken from the paper
factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]



# Weighted Scaled (Equalized Learning Rate) Convolutional Layer
class WSConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
		self.bias = self.conv.bias
		self.conv.bias = None
		# The authors of the paper found that to ensure healthy competition between the Gen & Disc
		#	it is essential that layers learn at a similar speed. To achieve this and Equalized
		#	Learning Rate is used. This is accomplished with the same formula as He Initialization.
		# The "scale" factor below is used to scale weights to ensure even learning EVERY forward pass
		self.scale = (gain / (in_channels * kernel_size**2)) ** 0.5

		# Initialize Conv Layer
		nn.init.normal_(self.conv.weight) #Initialize weights with a normal distribution
		nn.init.zeros_(self.bias) #Fill "bias" with zeros

	def forward(self, x):
		# Upon usage of WSConv2d, multiply input by scale constant
		return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1) #Reshape bias so it can be added to scaled conv of input



# Pixel Normalization (instead of BatchNorm)
class PixelNorm(nn.Module):
	def __init__(self):
		super().__init__()
		self.epsilon = 1e-8 #From paper

	# Pixel values are normalized to unit length
	def forward(self, x):
		# Dimension 1 in our data contains the RGB data we desire
		return (x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon))



# Block of Convolutional Layers
class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, use_pixelnorm=True): #Generator (not Disc) uses PixelNorm
		super().__init__()
		# Create a 3x3 (flexible) block from the paper
		self.conv1 = WSConv2d(in_channels, out_channels)
		self.conv2 = WSConv2d(out_channels, out_channels)
		self.leaky = nn.LeakyReLU(0.2)
		self.pn = PixelNorm()
		self.use_pn = use_pixelnorm

	def forward(self, x):
		# Employ 3x3 conv block, adjust PixelNorm for Gen vs Disc
		x = self.leaky(self.conv1(x))
		x = self.pn(x) if self.use_pn else x
		x = self.leaky(self.conv2(x))
		x = self.pn(x) if self.use_pn else x
		return x



# Generator Model
class Generator(nn.Module):
	def __init__(self, z_dim, in_channels, img_channels=3):
		super().__init__()
		self.initial = nn.Sequential(
			# This implementation does NOT include a WSConvTranspose (with an Equalized LR), for the portion below_________
			PixelNorm(),
			nn.ConvTranspose2d(in_channels=z_dim, out_channels=in_channels, kernel_size=4, stride=1, padding=0), #1x1 -> 4x4
			#______________________________________________________________________________________________________________
			nn.LeakyReLU(0.2),
			WSConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.2),
			PixelNorm()
		)

		# Both prog_blocks and rgb_layers are flexible Pytorch networks
		# Prog blocks are the Conv 3x3, 3x3 blocks, and rgb layers are the rgb layers between them (toRGB or fromRGB in the paper)
		self.initial_rgb = WSConv2d(in_channels=in_channels, out_channels=img_channels, kernel_size=1, stride=1, padding=0)
		self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])
		# Loop, building prog/rgb blocks/layers of size factors[i] -> factors[i+1]
		for i in range(len(factors)-1):
			# Calculate in and out channels
			conv_in_c = int(in_channels * factors[i])
			conv_out_c = int(in_channels * factors[i+1])
			# Build and append the associated conv & rgb networks
			self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
			self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

	def fade_in(self, alpha, upscaled, generated):
		# This is only going to provide an output, between -1 & 1
		# A piece of the new "fading-in" (generated) and old "fading-out" (upscaled) data
		return torch.tanh((alpha * generated) + ((1 - alpha) * upscaled))

	def forward(self, x, alpha, steps):
		out = self.initial(x) #4x4

		# If there are no steps, no new layers have to be faded in, no upsampling is required, simply return the input through the network
		if steps == 0:
			return self.initial_rgb(out)

		# If steps>0, iterate through prog_blocks using it (ie. passing data through a NN)
		for step in range(steps):
			# Each step, upsample the input image to increase its size 2X
			upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
			# Each step, iterate through prog blocks, passing the same image through
			out = self.prog_blocks[step](upscaled)

		# Wherever the loop ends, use that output to calculate the result through an RGB layer
		# RGB layer is the last layer from the paper (for the generator)
		final_upscaled = self.rgb_layers[steps-1](upscaled)
		final_out = self.rgb_layers[steps](out)

		# Fade-in between the 2nd to last and last image (using alpha) is returned
		return self.fade_in(alpha, final_upscaled, final_out)



# Discriminator Model (AKA Critic)
class Discriminator(nn.Module):
	def __init__(self, in_channels, img_channels=3):
		super().__init__()
		self.leaky = nn.LeakyReLU(0.2)

		# Both prog_blocks and rgb_layers are flexible Pytorch networks
		self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
		# Build their layers, working through the factors backwards
		for i in range(len(factors)-1, 0, -1):
			conv_in_c = int(in_channels * factors[i])
			conv_out_c = int(in_channels * factors[i-1])
			self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
			self.rgb_layers.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))

		# This is the final block, similar to the first block of the Generator,
		#	which does not have an existing class (4x4 -> 1x1)
		self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
		self.rgb_layers.append(self.initial_rgb)
		self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
		self.final_block = nn.Sequential(
				WSConv2d(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
				nn.LeakyReLU(0.2),
				WSConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
				nn.LeakyReLU(0.2),
				WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
			)

	def fade_in(self, alpha, downscaled, out):
		# A combination of the output and downscaled, scaled by an alpha parameter
		return ((alpha*out) + ((1-alpha)*downscaled))

	def minibatch_std(self, x):
		batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
		return torch.cat([x, batch_statistics], dim=1)

	def forward(self, x, alpha, steps):
		cur_step = len(self.prog_blocks) - steps #Steps in the Disc is from the end of the network!
		out = self.leaky(self.rgb_layers[cur_step](x))

		# If there are no steps, no new layers have to be added, no downsampling is required, simply
		#	pass the input through the existing network
		if steps == 0:
			out = self.minibatch_std(out)
			return self.final_block(out).view(out.shape[0], -1) #Adjust the shape

		# Else, downscale & fade in the image so it can enter the Discriminator
		downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x))) #Downsample the image
		out = self.avg_pool(self.prog_blocks[cur_step](out))
		out = self.fade_in(alpha, downscaled, out)

		# Then, pass the image through the appropriate (additional) layers using cur_step
		for step in range(cur_step+1, len(self.prog_blocks)):
			# Repeatedly 3x3 Block ("prog_blocks"), then downsample
			out = self.prog_blocks[step](out)
			out = self.avg_pool(out)

		# Pass through final block to get 4x4 resolution
		out = self.minibatch_std(out)
		return self.final_block(out).view(out.shape[0], -1)



# Test the code
if __name__ == "__main__":
	Z_DIM = 50
	IN_CHANNELS = 256
	gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3)
	critic = Discriminator(IN_CHANNELS, img_channels=3)

	# Run image sizes up to 1024x1024
	for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
		num_steps = int(log2(img_size/4)) #log2 as image size doubles per loop
		x = torch.randn((1, Z_DIM, 1, 1))
		z = gen(x, 0.5, steps=num_steps)
		assert z.shape == (1, 3, img_size, img_size)
		out = critic(z, alpha=0.5, steps=num_steps)
		assert out.shape == (1,1)
		print(f"Success! At img size: {img_size}")
