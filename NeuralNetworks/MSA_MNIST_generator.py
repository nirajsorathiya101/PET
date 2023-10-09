import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import NeuralNetworks
# this snippet fixes a bug in windows
if os.name == 'nt':  # checks if we are on a windows machine
    import ctypes
    ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


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

    mnist_testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(
        dataset=mnist_testset,
        batch_size=1,
        shuffle=True)

    net = NeuralNetworks.NetMNISTMSA().to(device)  # Transfers the model on gpu if available


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    EPOCHS = 20
    print(f'training target model for the model stealing attack for {EPOCHS} epochs')
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

    file__path = os.path.join(main_dir_path, 'NeuralNetworks/MSA_MNIST_net.pth')
    torch.save(net.state_dict(), file__path)

    # checking the performace of the model
    match = 0
    no_match = 0
    print('checking the correct and incorrect predictions of the model of the target model model:')
    for i, data in tqdm(enumerate(testloader, 0)):
        with torch.no_grad():
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)  # transfers the tensors on gpu
            outputs = net(inputs)
            _, index_output = torch.max(outputs, 1)
            if index_output == labels:
                match += 1
            else:
                no_match += 1
    print(
        f'correctly predicted : {match} incorrectly predicted : {no_match}')


if __name__ == '__main__':
    run()
