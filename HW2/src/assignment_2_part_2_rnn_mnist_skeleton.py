# -*- coding: utf-8 -*-
"""Assignment_2_Part_2_RNN_MNIST_vp1.ipynb
Overall structure:

1) Set Pytorch metada
- seed
- tensorflow output
- whether to transfer to gpu (cuda)

2) Import data
- download data
- create data loaders with batchsie, transforms, scaling

3) Define Model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop

# Step 1: Pytorch and Training Metadata
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt

from tqdm import tqdm

batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.001  # 0.01 0.001
optimizer_type = 'sgd'  # 'sgd' 'adam'
hidden_size = 64  # 64 128 256
try_cuda = True
seed = 1000
logging_interval = 10  # how many batches to wait before logging
logging_dir = None

INPUT_SIZE = 28

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("../") / Path(f"runs/")
    runs_dir.mkdir(exist_ok=True)

    logging_dir = runs_dir / Path(f"minst_{datetime_str}")

    logging_dir.mkdir(exist_ok=True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

# deciding whether to send to the cpu or not if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""

# Setting up data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.01307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

# plot one example
print(train_dataset.train_data.size())  # (60000, 28, 28)
print(train_dataset.train_labels.size())  # (60000)
plt.imshow(train_dataset.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_dataset.train_labels[0])
# plt.show()

"""# Step 3: Creating the Model"""


class Net(nn.Module):
    def __init__(self, base_model_type='rnn', input_size=INPUT_SIZE, hidden_size=128, num_layers=1, num_classes=10):
        super(Net, self).__init__()

        if base_model_type == 'rnn':
            self.base_model = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif base_model_type == 'lstm':
            self.base_model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif base_model_type == 'gru':
            self.base_model = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, hidden = self.base_model(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


model = Net(base_model_type='lstm', hidden_size=hidden_size).to(device)

if cuda:
    model.cuda()

if optimizer_type == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer_type == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

"""# Step 4: Train/Test"""


# Defining the test and trainig loops

def train(epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()
    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()

            data = data.view(-1, 28, 28)

            optimizer.zero_grad()
            output = model(data)  # forward
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if batch_idx % logging_interval == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #            100. * batch_idx / len(train_loader), loss.item()))

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
        data, target = data.cuda(), target.cuda()
        data = data.view(-1, 28, 28)

        output = model(data)
        test_loss += criterion(output, target).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))


# Training loop
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

    # save model
    model_name = f'model.pt'
    torch.save(model.state_dict(), logging_dir + "/" + model_name)
    # store params in yml file
    params = {
        "epochs": epochs,
        "lr": lr,
        "optimizer_type": optimizer_type,
        "hidden_size": hidden_size,
    }
    with open(logging_dir + "/" + "params.yml", "w") as outfile:
        yaml.dump(params, outfile, default_flow_style=False)

writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""
