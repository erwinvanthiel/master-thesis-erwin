import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import TargetClassifier, AuxiliaryClassifier, Discriminator, Generator, initialize_weights

# Hyperparameters.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 100000
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
NUM_CLASSES = 10

TARGET_CLASS = 7
ALPHA = 1
BETA = 1
GAMMA = 1

# Initialize models and params
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
aux = AuxiliaryClassifier(CHANNELS_IMG, NUM_CLASSES).to(device)
target = TargetClassifier(CHANNELS_IMG, NUM_CLASSES).to(device)

# initialize optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
aux_criterion = nn.CrossEntropyLoss()
target_criterion = nn.CrossEntropyLoss()

# Load and init model parameters
critic_state = torch.load("critic_model", map_location=device)
critic.load_state_dict(critic_state["state_dict"])

aux_state = torch.load("aux_model", map_location=device)
aux.load_state_dict(aux_state["state_dict"])

initialize_weights(gen)


# Code for constructing class conditional input noise
def generate_class_onehot(label, batch_size, number_of_classes, device='cpu'):
    class_onehot = torch.zeros((batch_size, number_of_classes, 1, 1), dtype=float).to(device)
    class_onehot[torch.arange(batch_size), torch.tensor(label).to(device)] = 1
    class_onehot = class_onehot.float()
    return class_onehot

def generate_class_conditional_noise(label, batch_size, number_of_classes, nz, device='cpu'):
    noise = torch.randn(batch_size, nz, 1, 1).to(device)
    class_onehot = generate_class_onehot(label, batch_size, number_of_classes, device)
    noise[np.arange(batch_size), :number_of_classes] = class_onehot[np.arange(batch_size)]
    return noise

# Setup tensorboard logging
writer_fake = SummaryWriter(f"logs/adversarials")
step = 0

# toggle model modes
gen.train()
aux.eval()
target.eval()
critic.eval()

# 1 epoch corresponds to 1 batch
for epoch in range(NUM_EPOCHS):

	# Generate fakes from a batch of noise
	labels = torch.randint(9, (BATCH_SIZE, 1)).reshape(-1).to(device)
	noise = generate_class_conditional_noise(labels, BATCH_SIZE, NUM_CLASSES, Z_DIM, device)
	fake = gen(noise).to(device)

	# Calculate losses
	disc_loss = -torch.mean(critic(fake).reshape(-1))
	aux_loss = aux_criterion(aux(fake), labels)
	target_loss = target_criterion(target(fake), torch.ones(BATCH_SIZE).long().to(device) * TARGET_CLASS)
	total_loss = ALPHA * disc_loss + BETA * aux_loss + GAMMA * target_loss

	# Perform gradient descent and update generator params
	gen.zero_grad()
	total_loss.backward()
	opt_gen.step()

	# Print losses occasionally and print to tensorboard
	if epoch % 100 == 0 and epoch > 0:
		print(
        	f"Epoch [{epoch}/{NUM_EPOCHS}] Loss D: {disc_loss:.4f}, loss aux: {aux_loss:.4f}, Loss classifier: {target_loss} Total Loss: {total_loss}"
        	)
		with torch.no_grad():
			# take out (up to) 32 examples
			img_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
			writer_fake.add_image("Fake", img_grid, global_step=step)
		step += 1