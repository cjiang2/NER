import os
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Parameters for pytorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Hyperparameters
si_c = 0.1
si_epsilon = 0.1

batch_size = 64
lr = 1e-3
epochs = 20    # Epochs per task

# Permuted MNIST Settings
# Generate the tasks specifications as a list of random permutations of the input pixels
n_tasks = 10
tasks_permutation = []
for _ in range(n_tasks):
	tasks_permutation.append(np.random.permutation(784))

class Net(nn.Module):
    def __init__(self, units=512):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trainer():
    def __init__(self, model, train_loader, test_loader, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.params_name = [n for n, p in self.model.named_parameters() if p.requires_grad]

        # Initialize importance measure w and omega as zeros, record previous weights
        self.w = {}
        self.omega = {}
        self.prev_params = {}
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                self.prev_params[n] = p.clone().detach()
                self.w[n] = torch.zeros(p.shape).float().to(device)
                self.omega[n] = torch.zeros(p.shape).float().to(device)

    def full_cycle(self):
        # Training per task
        for task in range(n_tasks):
            print('Training task {}'.format(task))

            # Training
            self.model.train()
            for epoch in range(epochs):
                if epoch % 5==0:
                    print('Epoch {}'.format(epoch))
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data = data.view(-1, 784)[:, tasks_permutation[task]] # Permutate mnist image by task permutation
                    self.train_sample(data, target)
            
            # Update omega and the referene weights, set parameter importance to 0
            self.on_update()
            
            # Evaluation by task
            self.model.eval()
            for task_test in range(task+1):
                test_loss = 0
                correct = 0.0
                with torch.no_grad():
                    for data, target in self.test_loader:
                        data = data.view(-1, 784)[:, tasks_permutation[task]]

                        data, target = data.to(device), target.to(device)
                        output = self.model(data)

                        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(self.test_loader.dataset)

                print('Test set: Average loss: {:.4f}, Accuracy: {})'.format(test_loss, 
                                                                             correct / len(self.test_loader.dataset)))
            print()

        return

    def train_sample(self, data, target):
        # Forward
        data, target = data.to(device), target.to(device)
        output = self.model(data)

        # Collect unregularized gradients
        unreg_grads = {}
        unreg_loss = F.cross_entropy(output, target)
        self.optimizer.zero_grad()
        unreg_loss.backward(retain_graph=True)
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                unreg_grads[n] = p.grad.clone().detach()
        
        # Calculate surrogate loss
        surrogate_loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                surrogate_loss += torch.sum(self.omega[n]*((p.detach() - self.prev_params[n])**2))
        loss = F.cross_entropy(output, target) + si_c*surrogate_loss

        # One train step with surrogate loss now
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update importance right after every train step
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                delta = p.detach() - self.prev_params[n]
                self.w[n] = self.w[n] - unreg_grads[n]*delta

    def on_update(self):
        # Calculate regularization strength
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                self.omega[n] += self.w[n] / ((p.detach() - self.prev_params[n])**2 + si_epsilon)
                self.omega[n] = F.relu(self.omega[n])

                # Reset importance measure and record previous weights
                self.w[n] = self.w[n]*0.0
                self.prev_params[n] = p.clone().detach()

def main():
    # Data loaders
    kwargs = {'num_workers': 1 if os.name is 'nt' else multiprocessing.cpu_count(), 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True, **kwargs)
        
    # Models, optimizer and trainer
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    si_trainer = Trainer(model=model, 
                         train_loader=train_loader,
                         test_loader=test_loader,
                         optimizer=optimizer)
    si_trainer.full_cycle()

    return

if __name__ == '__main__':
    main()