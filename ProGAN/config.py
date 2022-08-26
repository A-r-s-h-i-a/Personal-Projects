import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 4
IMAGE_SIZE = 512
DATASET = "celeb_hq"
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [128, 128, 128, 64, 16, 8, 4, 2, 1] #Change depending on VRAM
CHANNELS_IMG = 3
Z_DIM = 256 #Paper used 512, I'm using 256 for less VRAM (faster training
IN_CHANNELS = 256 #Paper used 512, I'm using 256 for less VRAM (faster training)
LAMBDA_GP = 10
# Change epochs depending on dataset and training features/desires
PROGRESSIVE_EPOCHS = [20] * len(BATCH_SIZES) #Can change N in [N]*len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4