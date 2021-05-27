"""
Training of WGAN-GP
"""
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
from model import AuxiliaryClassifier, Discriminator, Generator, initialize_weights


# Toggles for loading and saving model parameters
save_model = True
load_model = False

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
NUM_CLASSES = 10
AUX_MULTIPLYER = 1

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

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
aux = AuxiliaryClassifier(CHANNELS_IMG, NUM_CLASSES).to(device)
initialize_weights(gen)
initialize_weights(critic)
initialize_weights(aux)


# initialize optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_aux = optim.Adam(aux.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
aux_criterion = nn.CrossEntropyLoss()

# Load model parameters if desired
if load_model:
        critic_state = torch.load("critic_model", map_location=device)
        critic.load_state_dict(critic_state["state_dict"])

        gen_state = torch.load("gen_model", map_location=device)
        gen.load_state_dict(gen_state["state_dict"])

        aux_state = torch.load("aux_model", map_location=device)
        aux.load_state_dict(aux_state["state_dict"])


def generate_class_onehot(label, batch_size, number_of_classes, device='cpu'):
    class_onehot = torch.zeros((batch_size, number_of_classes, 1, 1), dtype=float).to(device)
    class_onehot[torch.arange(batch_size), torch.tensor(label).to(device)] = 1
    class_onehot = class_onehot.float()
    return class_onehot

def generate_class_conditional_noise(label, batch_size, number_of_classes, nz, device='cpu'):
    noise = torch.randn(batch_size, nz, 1, 1).to(device)
    class_onehot = generate_class_onehot(label, batch_size, number_of_classes, device)
    noise[np.arange(batch_size), :number_of_classes] = class_onehot[np.arange(batch_size)]
    # return torch.randn(batch_size, nz, 1, 1).to(device)
    return noise


# for tensorboard plotting
# fixed_labels = np.arange(32) % 10
# fixed_noise = generate_class_conditional_noise(fixed_labels, 32, NUM_CLASSES, Z_DIM, device)
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

gen.train()
critic.train()
aux.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, label) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        class_onehot = generate_class_onehot(label, cur_batch_size, NUM_CLASSES, device)
        label = label.to(device)
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):

            noise = generate_class_conditional_noise(label, cur_batch_size, NUM_CLASSES, Z_DIM, device)
            
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            aux_real = aux(real).to(device)
            aux_fake = aux(fake)
            aux_loss = (aux_criterion(aux_real, label) + aux_criterion(aux_fake, label))
            aux.zero_grad()
            aux_loss.backward(retain_graph=True)
            opt_aux.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        
        aux_fake = aux(fake)
        loss_aux = aux_criterion(aux_fake, label)

        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        
        total_gen_loss = loss_gen + AUX_MULTIPLYER * loss_aux
        gen.zero_grad()
        total_gen_loss.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {total_gen_loss:.4f}, Loss classifier: {aux_loss}"
            )

            with torch.no_grad():
                fake = gen(noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1


    if save_model:
        print("Saving model parameters...")

        critic_state = {
            "state_dict": critic.state_dict(),
            "optimizer": opt_critic.state_dict(),
        }
        torch.save(critic_state, "critic_model")

        gen_state = {
            "state_dict": gen.state_dict(),
            "optimizer": opt_gen.state_dict(),
        }
        torch.save(gen_state, "gen_model")

        aux_state = {
            "state_dict": aux.state_dict(),
            "optimizer": opt_aux.state_dict(),
        }
        torch.save(aux_state, "aux_model")

