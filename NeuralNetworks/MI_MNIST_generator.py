import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import NeuralNetworks
# this snippet fixes a bug in windows
if os.name == 'nt':  # checks if we are on a windows machine
    import ctypes
    ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].


def run():

    device = torch.device('cpu') # default is cpu
    # this checks if a cuda gpu is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('yes gpu')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    main_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(main_dir_path, 'Training_data/data_MNIST')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    batch_size = 64

    mnist_trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        dataset=mnist_trainset,
        batch_size=batch_size,
        shuffle=False)


    net = NeuralNetworks.NetMNIST().to(device)  # Transfers the model on gpu if available

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    EPOCHS = 5
    print(f'training for {EPOCHS} epochs')
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        for i, data in tqdm(enumerate(trainloader, 0)):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # transfers the tensors on gpu
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print('Finished Training')

    file__path = os.path.join(main_dir_path, 'NeuralNetworks/MI_MNIST_net.pth')
    torch.save(net.state_dict(), file__path)


if __name__ == '__main__':
    run()
