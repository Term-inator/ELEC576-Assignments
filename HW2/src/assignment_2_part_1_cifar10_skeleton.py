# -*- coding: utf-8 -*-
"""Assignment_2_Part_1_Cifar10_vp1.ipynb

Purpose: Implement image classsification nn the cifar10
dataset using a pytorch implementation of a CNN architecture (LeNet5)

Pseudocode:
1) Set Pytorch metada
- seed
- tensorboard output (logging)
- whether to transfer to gpu (cuda)

2) Import the data
- download the data
- create the pytorch datasets
    scaling
- create pytorch dataloaders
    transforms
    batch size

3) Define the model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates
        f. Calculate accuracy, other stats
    - Test:
        a. Calculate loss, accuracy, other stats

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop




"""

# Step 1: Pytorch and Training Metadata

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import yaml
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

from tqdm import tqdm
import json

# hyperparameters
batch_size = 64             # 32 64
epochs = 5
lr = 0.001                  # 0.001 0.01
optimizer_type = "adam"     # "adam" "sgd"
momentum = 0.9              # 0.9 0.5
try_cuda = True
seed = 1000

# Architecture
num_classes = 10

# otherum
logging_interval = 10  # how many batches to wait before logging
logging_dir = None
grayscale = True

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("../") / Path(f"runs/")
    runs_dir.mkdir(exist_ok=True)

    logging_dir = runs_dir / Path(f"{datetime_str}")

    logging_dir.mkdir(exist_ok=True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

# deciding whether to send to the cpu or not if available
device = 'cuda' if torch.cuda.is_available() and try_cuda else 'cpu'
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""

# downloading the cifar10 dataset

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# download and transform cifar10 training data
train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def check_data_loader_dim(loader):
    # Checking the dataset
    for images, labels in loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break


check_data_loader_dim(train_loader)
check_data_loader_dim(test_loader)

"""# 3) Creating the Model"""

layer_1_n_filters = 32
layer_2_n_filters = 64
fc_1_n_nodes = 120  # 1024 120
padding = "same"
kernel_size = 5
verbose = False

# calculating the side length of the final activation maps
final_length = 32 // 2 // 2

if verbose:
    print(f"final_length = {final_length}")


class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, layer_1_n_filters, kernel_size, padding=padding),  # 32x32x1 ->32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32x32 -> 16x16x32
            nn.Conv2d(layer_1_n_filters, layer_2_n_filters, kernel_size, padding=padding),  # 16x16x32 -> 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16x64 -> 8x8x64
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_length * final_length * layer_2_n_filters * in_channels, fc_1_n_nodes),
            nn.Tanh(),
            nn.Linear(fc_1_n_nodes, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.classifier[0].in_features)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


model = LeNet5(num_classes, grayscale)

if cuda:
    model.cuda()

if optimizer_type == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer_type == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

"""# Step 4: Train/Test Loop"""


# Defining the test and trainig loops

def train(epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()
    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits, probas = model(data)  # forward

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            if batch_idx % logging_interval == 0:
                n_iter = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', loss.item(), n_iter)

            pbar.set_description(f"Epoch {epoch}")
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(size_average=False)
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        logits, probas = model(data)

        # [insert-code: finish testing loop and logging metrics]
        test_loss += criterion(logits, target).item()
        pred = logits.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))


for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

    # save model
    model_name = f'cifar10_lenet5.pt'
    torch.save(model.state_dict(), logging_dir + "/" + model_name)
    # store params in yml file
    params = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "optimizer_type": optimizer_type,
        "momentum": momentum if optimizer_type == "sgd" else None,
        "fc_1_n_nodes": fc_1_n_nodes,
    }
    with open(logging_dir + "/" + "params.yml", "w") as outfile:
        yaml.dump(params, outfile, default_flow_style=False)


writer.close()

# Visualize
conv1_weights = model.features[0].weight.data.clone().cpu()

conv1_weights -= conv1_weights.min()
conv1_weights /= conv1_weights.max()

# plot the weights
fig = plt.figure(figsize=(8, 8))
for i in range(32):
    ax = fig.add_subplot(4, 8, i + 1)
    ax.imshow(conv1_weights[i, 0, :, :])
    ax.axis('off')
    ax.set_title(str(i+1))
plt.savefig(f"{logging_dir}/conv1_weights.png")

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""
