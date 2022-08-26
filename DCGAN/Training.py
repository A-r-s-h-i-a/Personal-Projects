# Training the DCGAN on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from Model import Discriminator, Generator, initialize_weights



LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64



# Define how transforms will process an image passed to it,
#	each processing step is defined in order
transforms = transforms.Compose(
	[
		transforms.Resize(IMAGE_SIZE), # Resize image
		transforms.ToTensor(), # Convert it to a tensor
		transforms.Normalize( # Normalize it
			[0.5 for _ in range(CHANNELS_IMG)],
			[0.5 for _ in range(CHANNELS_IMG)]
		)
	]
)

# Download the dataset, and create loader to get specific datapoints
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the Discriminator and Generator
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC)
initialize_weights(disc)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)
initialize_weights(gen)
fixed_noise = torch.randn(32, Z_DIM, 1, 1)

# Setup training tools, optimizer & loss function
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Set models to training mode so BatchNorm layers use per-batch stats
# and Dropout layers are activated (they should be by default)
gen.train()
disc.train()

# BEGIN TRAINING!!!
for epoch in range(EPOCHS):
	for batch_idx, (real, _) in enumerate(loader): #ignore labels
		print(batch_idx)
		noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1))
		fake = gen(noise)

		# Train Discriminator to maximize [log(D(x)) + log(1-D(G(z)))]
		disc_real = disc(real).reshape(-1)
		loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
		disc_fake = disc(fake).reshape(-1)
		loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
		loss_disc = (loss_disc_real + loss_disc_fake) / 2
		disc.zero_grad()
		loss_disc.backward(retain_graph=True)
		opt_disc.step()

		# Train Generator to maximize log(D(G(z))
		output = disc(fake).reshape(-1)
		loss_gen = criterion(output, torch.ones_like(output))
		gen.zero_grad()
		loss_gen.backward()
		opt_gen.step()

		if batch_idx % 50 == 0:
			# Display per-Epoch Results
			with torch.no_grad():
				print(
					f"Epoch [{epoch}/{EPOCHS}]  -  Batch [{batch_idx}/{len(loader)}]\t",
					f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
				)
				fake = gen(fixed_noise)

				fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(10,10))
				loc = 0
				for ix in range(len(axes)):
					for iy in range(len(axes[0])):
						axes[ix, iy].imshow(fake[loc][0])
						loc += 1

				fig.tight_layout()
				plt.savefig('Results/Epoch'+str(epoch)+'_Batch'+str(batch_idx)+'.png')
