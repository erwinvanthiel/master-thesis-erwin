import numpy as np
from fid import calculate_fid
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import TargetClassifier, AuxiliaryClassifier, Generator, initialize_weights

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 100000
FEATURES_GEN = 16
NUM_CLASSES = 10
TARGET_CLASS = 7
ALPHA = 1
BETA = 1
GAMMA = 0


# Get MNIST dataset
transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# Initialize models and params
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
aux = AuxiliaryClassifier(CHANNELS_IMG, NUM_CLASSES).to(device)
target = TargetClassifier(CHANNELS_IMG, NUM_CLASSES).to(device)


# Initialize optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
aux_criterion = nn.CrossEntropyLoss()
target_criterion = nn.CrossEntropyLoss()


# Load and init model parameters
aux_state = torch.load("aux_model", map_location=device)
aux.load_state_dict(aux_state["state_dict"])

target_state = torch.load("target_model", map_location=device)
target.load_state_dict(target_state["state_dict"])

gen_state = torch.load("gen", map_location=device)
gen.load_state_dict(gen_state["state_dict"])

feature_extractor = nn.Sequential(aux.conv1, aux.pool, aux.conv2)


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


# Configure model modes
gen.train()
aux.eval()
target.eval()


# TRAINING LOOP STARTS HERE
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, label) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        label = label.to(device)


        # Generate fakes from a batch of noise
        noise = generate_class_conditional_noise(label, cur_batch_size, NUM_CLASSES, Z_DIM, device)
        fake = gen(noise).to(device)


        # Calculate losses

        # FID distance
        fake_activation = feature_extractor(fake).reshape(-1).cpu().detach().numpy()
        real_activation = feature_extractor(real).reshape(-1).cpu().detach().numpy()
        fid_loss = calculate_fid(fake_activation, real_activation)

        # Auxiliary classifier loss
        aux_loss = aux_criterion(aux(fake), label)

        # Adversarial loss
        target_loss = target_criterion(target(fake), torch.ones(cur_batch_size).long().to(device) * TARGET_CLASS)

        # Weighted sum of the above, a.k.a total loss
        total_loss = ALPHA * fid_loss + BETA * aux_loss + GAMMA * target_loss


        # Perform gradient descent and update generator params
        gen.zero_grad()
        total_loss.backward()
        opt_gen.step()


        # Print losses occasionally and print to tensorboard
        if epoch % 100 == 0 and epoch > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Loss FID: {fid_loss:.4f}, loss aux: {aux_loss:.4f}, Loss classifier: {target_loss} Total Loss: {total_loss}"
                )
            with torch.no_grad():
                # take out (up to) 32 examples
                img_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_fake.add_image("Fake", img_grid, global_step=step)
            step += 1