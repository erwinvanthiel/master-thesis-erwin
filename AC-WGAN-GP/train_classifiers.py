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


# Toggles for loading and saving model parameters
save_model = True
load_model = False

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_EPOCHS = 100
LAMBDA_GP = 10
NUM_CLASSES = 10

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
# comment mnist above and uncomment below for training on CelebA
#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

aux = AuxiliaryClassifier(CHANNELS_IMG, NUM_CLASSES).to(device)
target = TargetClassifier(CHANNELS_IMG, NUM_CLASSES).to(device)

opt_target = optim.Adam(target.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_aux = optim.Adam(aux.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

initialize_weights(target)
initialize_weights(aux)

aux_criterion = nn.CrossEntropyLoss()
target_criterion = nn.CrossEntropyLoss()


for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, label) in enumerate(loader):

    	aux_loss = aux_criterion(aux(data), label)
    	aux.zero_grad()
    	aux_loss.backward(retain_graph=True)
    	opt_aux.step()

    	target_loss = target_criterion(target(data), label)
    	target.zero_grad()
    	target_loss.backward(retain_graph=True)
    	opt_target.step()

    	print(aux_loss, target_loss)

    print("Epoch finished, saving model parameters...")

    aux_state = {
	  "state_dict": aux.state_dict(),
	  "optimizer": opt_aux.state_dict(),
    }
    torch.save(aux_state, "aux_model")
    target_state = {
    "state_dict": target.state_dict(),
    "optimizer": opt_target.state_dict(),
    }
    torch.save(target_state, "target_model")