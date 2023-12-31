# -*- coding: utf-8 -*-
"""Assignment_1_Pytorch_MNIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i9KpbQyFU4zfq8zLLns8a2Kd8PRMGsaZ

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

Acknowledgments:https://github.com/motokimura/pytorch_tensorboard/blob/master/main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.01
try_cuda = True
seed = 1000
logging_interval = 10 # how many batches to wait before logging
logging_dir = './logs'

actFun_type = 'relu'
optim_type = 'Adagrad'

# 1) setting up the logging
writer = SummaryWriter(logging_dir)

#deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    device = 'cuda'
    torch.cuda.manual_seed(seed)
else:
    device = 'cpu'
    torch.manual_seed(seed)

# Setting up data
transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.01307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                    transform=transform),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transform),
    batch_size=test_batch_size, shuffle=True)


def actFun(x, type='relu'):
    if type == 'relu':
        return F.relu(x)
    elif type == 'tanh':
        return F.tanh(x)
    elif type == 'sigmoid':
        return F.sigmoid(x)
    elif type == 'leaky_relu':
        return F.leaky_relu(x)
    else:
        raise Exception('Unknown actFun type')


# Defining Architecture,loss and optimizer
class Net(nn.Module):
    def __init__(self, actFun_type='relu'):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.maxpool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(320, 50)
        self.fc1_drop = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(50, 10)

        self.actFun_type = actFun_type

    def forward(self, x):

        x = self.conv1(x)
        x = actFun(x, self.actFun_type)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = actFun(x, self.actFun_type)
        x = self.maxpool2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = actFun(x, self.actFun_type)
        x = self.fc1_drop(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x


model = Net().to(device)

def get_optimizer(optim_type, model, lr):
    if optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optim_type == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optim_type == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise Exception('Unknown optimizer type')

    return optimizer

optimizer = get_optimizer(optim_type, model, lr)


# Defining the test and trainig loops
eps=1e-13

def train(epoch):
    model.train()

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    n_iter = (epoch - 1) * len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(torch.log(output+eps), target) # = sum_k(-t_k * log(y_k))
        loss.backward()
        optimizer.step()

        if batch_idx % logging_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar('Train/Loss', loss.item(), n_iter)

            for name, layer in model.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    writer.add_histogram(f'{name}.net_inputs', layer.weight.grad, global_step=n_iter)
                    writer.add_histogram(f'{name}.weights', layer.weight, global_step=n_iter)
                    writer.add_histogram(f'{name}.biases', layer.bias, global_step=n_iter)

            with torch.no_grad():
                activations = data
                for name, layer in model.named_modules():
                    if name == '':
                        continue
                    if name == 'conv1' or name == 'conv2':
                        activations = layer(activations)
                        activations = actFun(activations, type=actFun_type)
                        writer.add_histogram(f'{name}.activations_conv_{actFun_type}', activations, global_step=n_iter)
                    elif name == 'maxpool1' or name == 'maxpool2':
                        activations = layer(activations)
                        writer.add_histogram(f'{name}.activations_maxpool', activations, global_step=n_iter)
                    elif name == 'fc1':
                        activations = activations.view(-1, 320)
                        activations = layer(activations)
                        activations = actFun(activations, type=actFun_type)
                        writer.add_histogram(f'{name}.activations_fc1_{actFun_type}', activations, global_step=n_iter)
                    else:
                        activations = layer(activations)

        n_iter += 1

    # Log model parameters to TensorBoard at every epoch
    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram(
            f'{layer}/{attr}',
            param.clone().cpu().data.numpy(),
            n_iter)



def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    #criterion = nn.CrossEntropyLoss()

    #criterion = nn.CrossEntropyLoss(size_average = False)
    criterion = nn.NLLLoss(size_average = False)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)

        test_loss += criterion(torch.log(output+eps), target,).item() # sum up batch loss (later, averaged over all test samples)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item() # sum up correct predictions

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    # print the performance
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_accuracy))

    # Log test/loss and test/accuracy to TensorBoard at every epoch
    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

# Training loop

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""