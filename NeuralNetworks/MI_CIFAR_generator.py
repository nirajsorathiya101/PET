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

def run():

    device = torch.device('cpu') # default is cpu
    # this checks if a cuda gpu is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('yes gpu')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    main_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(main_dir_path, 'Training_data/data')

    transform = transforms.Compose(
        [ transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    #images are 3 * 32 *32 tensors
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    #train loader is an iterable [sample_id, [multiple_sample_image_tensor, corresponding_validation_tensors]]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2, pin_memory=True)


    net = NeuralNetworks.NetCIFAR().to(device) # transfers the model on gpu

    # Let’s use a Classification Cross-Entropy loss and SGD with momentum.

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    EPOCHS = 5
    print(f'training for {EPOCHS} epochs')

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # transfers the tensors on gpu

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


    file__path = os.path.join(main_dir_path, 'NeuralNetworks/MI_CIFAR_net.pth')
    torch.save(net.state_dict(), file__path)


if __name__ == '__main__':

    run()
