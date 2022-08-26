import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from math import log2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import (
	gradient_penalty,
	plot_to_tensorboard,
	save_checkpoint,
	load_checkpoint,
	generate_examples
)
from model import Discriminator, Generator
import config



torch.backends.cudnn.benchmarks = True #Flag which will give us performance benefits

# A flexible loader for images of different sizes
def get_loader(image_size):
	transform = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.Normalize(
				[0.5 for _ in range(config.CHANNELS_IMG)],
				[0.5 for _ in range(config.CHANNELS_IMG)],
			)
		]
	)
	batch_size = config.BATCH_SIZES[int(log2(image_size/4))]
	dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
	loader = DataLoader(
		dataset,
		batch_size = batch_size,
		shuffle = True,
		num_workers = config.NUM_WORKERS,
		pin_memory=True
	)

	return loader, dataset



# A function to train networks of different sizes
def train_fn(	critic,
				gen,
				loader,
				dataset,
				step,
				alpha,
				opt_critic,
				opt_gen,
				tensorboard_step,
				writer,
				scaler_gen,
				scaler_critic,
				epoch
			):

	loop = tqdm(loader, leave=True)
	for batch_idx, (real, _) in enumerate(loop):
		real = real.to(config.DEVICE)
		cur_batch_size = real.shape[0]

		# Train Critic: maximize ->   (E[critic(real)] - E[critic(fake)])
		noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
		with torch.cuda.amp.autocast():
			fake = gen(noise, alpha, step)
			critic_real = critic(real, alpha, step)
			critic_fake = critic(fake.detach(), alpha, step) #Detach so we can reuse it
			gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
			# Make minimization problem by adding negative sign
			loss_critic = (
				- (torch.mean(critic_real) - torch.mean(critic_fake))
				+ config.LAMBDA_GP * gp
				+ (0.001 * torch.mean(critic_real ** 2))
			)
		opt_critic.zero_grad()
		scaler_critic.scale(loss_critic).backward()
		scaler_critic.step(opt_critic)
		scaler_critic.update()

		# Train Generator: maximize -> E[critic(gen_fake)]
		with torch.cuda.amp.autocast():
			gen_fake = critic(fake, alpha, step)
			loss_gen = -torch.mean(gen_fake)
		opt_gen.zero_grad()
		scaler_gen.scale(loss_gen).backward()
		scaler_gen.step(opt_gen)
		scaler_gen.update()

		# Update alpha, and ensure less than 1
		alpha += cur_batch_size / (len(dataset) * (config.PROGRESSIVE_EPOCHS[step]*0.5))
		alpha = min(alpha, 1)

		# On 500th loop, add to tensorboard
		if batch_idx % 500 == 0:
			with torch.no_grad():
				fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
			plot_to_tensorboard(
				writer,
				loss_critic.item(),
				loss_gen.item(),
				real.detach(),
				fixed_fakes.detach(),
				tensorboard_step,
			)
			tensorboard_step += 1
		
		# Add the following to tqdm to progress bar
		loop.set_postfix(
			res=str(2**(step+1))+'x'+str(2**(step+1)),
			epoch=epoch
		)

	return tensorboard_step, alpha



# Main function putting everything together
def main():
	# Initialize the Generator and Critic
	gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
	critic = Discriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)

	# Setup training tools
	opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
	opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
	scaler_gen = torch.cuda.amp.GradScaler()
	scaler_critic = torch.cuda.amp.GradScaler()

	# Setup writer for tensorboard plotting
	writer = SummaryWriter(f"logs/gan1")

	# IF LOAD MODEL - load the models
	if config.LOAD_MODEL:
		load_checkpoint(
			config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE
		)
		load_checkpoint(
			config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE
		)

	# Train the models
	gen.train()
	critic.train()

	tensorboard_step = 0
	# Start at the step that corresponds to starting image size #4->0, 8->1, 16->2, etc.
	step = int(log2(config.START_TRAIN_AT_IMG_SIZE/4))
	# Loop through different image resolutions
	for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
		alpha = 1e-5
		loader, dataset = get_loader(4*2**step)
		print(f"Image size: {4*2**step}")

		# Loop through epochs at each resolution, training
		for epoch in range(num_epochs):
			print(f"Epoch[{epoch+1}/{num_epochs}]")
			tensorboard_step, alpha = train_fn(
				critic,
				gen,
				loader,
				dataset,
				step,
				alpha,
				opt_critic,
				opt_gen,
				tensorboard_step,
				writer,
				scaler_gen,
				scaler_critic,
				epoch
			)

			# Save the model every epoch
			if config.SAVE_MODEL:
				save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
				save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)

		step += 1 #Progress to the next image size!



if __name__ == "__main__":
	main()