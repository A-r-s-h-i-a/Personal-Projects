import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.utils import save_image
from utils import USE_GPU, SAVE_IMG_PATH, NUM_IMAGES, MODEL



def generate_image(folder_path=SAVE_IMG_PATH, num_images=1, plot_image=False):
	# load the pre-trained ProGAN model
	model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub",
		"PGAN", model_name=MODEL, pretrained=True, #"celeba" for 128p and "celebAHQ-512" for 512p
		useGPU=USE_GPU)
	# sample random noise vectors
	(noise, _) = model.buildNoiseData(num_images)
	# pass the sampled noise vectors through the pre-trained generator
	with torch.no_grad():
		generatedImages = model.test(noise)
	# visualize the generated images
	grid = torchvision.utils.make_grid(
		generatedImages.clamp(min=-1, max=1), nrow=num_images,
		scale_each=True, normalize=True)

	if plot_image == True:
		plt.figure(figsize = (5,5))
		plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
		plt.show()

	return generatedImages



if __name__ == "__main__":
	for i in range(NUM_IMAGES):
		transform = T.Resize((256, 256))
		img = generate_image()
		resized_img = transform(img)

		# save generated image visualizations
		SAVE_IMG_PATH_i = SAVE_IMG_PATH + "/" + str(i) + ".jpeg"
		save_image(resized_img.clamp(min=-1, max=1),
			SAVE_IMG_PATH_i, nrow=NUM_IMAGES, scale_each=True,
			normalize=True)