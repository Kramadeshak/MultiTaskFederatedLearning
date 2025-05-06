import torch
import numpy as np


class FLModel(torch.nn.Module):
    def __init__(self, device):
        super(FLModel, self).__init__()
        self.optim = None
        self.loss_fn = None
        self.bn_layers = []
        self.device = device

    def set_optim(self, optim, init_optim=True):
        self.optim = optim
        if init_optim:
            self.empty_step()

    def empty_step(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def calc_acc(self, logits, y):
        raise NotImplementedError

    def train_step(self, x, y):
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.calc_acc(logits, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item(), acc

class MNISTModel(FLModel):
    def __init__(self, device):
        super(MNISTModel, self).__init__(device)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.fc0 = torch.nn.Linear(784, 200).to(device)
        self.relu0 = torch.nn.ReLU().to(device)
        self.fc1 = torch.nn.Linear(200, 200).to(device)
        self.relu1 = torch.nn.ReLU().to(device)

        self.out = torch.nn.Linear(200, 10).to(device)
        self.bn0 = torch.nn.BatchNorm1d(200).to(device)
        self.bn_layers = [ self.bn0 ]

    def forward(self, x):
        a = self.bn0(self.relu0(self.fc0(x)))
        b = self.relu1(self.fc1(a))

        return self.out(b)

    def calc_acc(self, logits, y):
        return (torch.argmax(logits, dim=1) == y).float().mean()
