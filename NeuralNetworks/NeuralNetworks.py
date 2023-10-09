import torch
import torch.nn as nn
import torch.nn.functional as F


class NetMNIST(nn.Module):
    def __init__(self):
        super(NetMNIST, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


class NetCIFAR(nn.Module):
    def __init__(self):
        super(NetCIFAR, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetCIFARCONV(nn.Module):

    def __init__(self):
        super(NetCIFARCONV, self).__init__()
        # 3*32*32
        self.conv1 = nn.Conv2d(3, 6, 5)        # 28 * 28 (h*w )  * 6
        self.pool = nn.MaxPool2d(2, 2)          # 14 * 14  (h*w )  *6
        self.conv2 = nn.Conv2d(6, 16, 5)        #(I – F + 2P)/S + 1
        self.pool = nn.MaxPool2d(2, 2)          # 14/2 * 14/2   * 16
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Tanh()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        x = F.relu(self.fc2(x))
        x = self.fc4(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return F.softmax(x, dim=1)


class NetMIAAttack(nn.Module):
    def __init__(self):
        super(NetMIAAttack, self).__init__()
        self.fc1 = nn.Linear(3, 9)
        self.fc2 = nn.Linear(9, 2)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return x


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):

        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetCIFARMSA(nn.Module):
    def __init__(self):
        super(NetCIFARMSA, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.tanh(x)
        x = self.fc3(x)
        return self.softmax(x)


class NetMNISTMSA(nn.Module):
    def __init__(self):
        super(NetMNISTMSA, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
