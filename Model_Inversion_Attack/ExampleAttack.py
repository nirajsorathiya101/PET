import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
from NeuralNetworks import NeuralNetworks
import InversionAttack


def run():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(dir_path, 'MIA_attack_model.pth')


    if sys.argv[1] == 'MNIST':

        # I'm extracting an image here for comparison, this is not passed to to the inversion attack
        path = os.path.join(root_dir_path, 'NeuralNetworks/MI_MNIST_net.pth')
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        mnist_trainset = torchvision.datasets.MNIST(root=os.path.join(root_dir_path,  'Training_data/data_MNIST'),
                                                    train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            dataset=mnist_trainset,
            batch_size=10,
            shuffle=True)

        model = NeuralNetworks.NetMNIST()

        iterator = iter(trainloader)
        images, labels = iterator.next()
        target = labels[0].item()
        image = images[0]
        plt.imshow(image[0].numpy(), cmap='gray')

    if sys.argv[1] == 'CIFAR':

        # I'm extracting an image here for comparison, this is not passed to to the inversion attack
        path = os.path.join(root_dir_path, 'NeuralNetworks/MI_CIFAR_net.pth')
        classes = ['plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # changing the images from the CIFAR dataset to grayscale
        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
             transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)])

        trainset = torchvision.datasets.CIFAR10(root=os.path.join(root_dir_path, 'Training_data/data'),
                                                train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=True, num_workers=2)
        model = NeuralNetworks.NetCIFAR()
        iterator = iter(trainloader)
        images, labels = iterator.next()
        target = labels[0]
        image = images[0]
        plt.imshow(image[0].numpy(), cmap='gray')

    try:
        plt.savefig(os.path.join(dir_path, 'test.png'), dpi=50)
        print(f'The label to invert is a {str(classes[target])} and an example is saved under test.png')
    except IOError:
        print(f"Unexpected error: {sys.exc_info()[0]}")

    model.load_state_dict(torch.load(path))  # Loading the neural network
    model.eval()  # Initialising the neural network
    example = InversionAttack.InversionAttack(model, image.unsqueeze_(1), classes, target)  # Creating an inversion attack object
    result = example.miface()  # this returns our image
    plt.imshow(np.squeeze(result.numpy()), aspect='equal', cmap='gray')  # plotting image

    try:
        plt.savefig(os.path.join(dir_path, 'result.png'), dpi=50)
        print(f'The result of the inversion attack on label {str(classes[target])} is saved under result.png')
    except IOError:
        print(f"Unexpected error: {sys.exc_info()[0]}")


if __name__ == '__main__':
    run()
