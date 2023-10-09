import os, sys
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
from Model_Stealing_Attack.ModelStealingAttack import ModelStealingAttack
from NeuralNetworks import NeuralNetworks
import torch
import torchvision
import torchvision.transforms as transforms
import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the attack model')
    opt = parser.parse_args()
    trainModel = opt.train

    if trainModel:
        attack = ModelStealingAttack()
        attack.MSAAttack()

    device = 'cpu'
    root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # loading both the target and the stolen model
    target_model = NeuralNetworks.NetMNISTMSA().to(device)
    try:
        data_path = os.path.join(root_dir_path, 'NeuralNetworks/MSA_MNIST_net.pth')
    except FileNotFoundError:
        print('You need to train the target model first with MSA_MNIST_generator.py')
    target_model.load_state_dict(torch.load(data_path))
    target_model.eval()

    attack_model = NeuralNetworks.NetMNISTMSA().to(device)
    try:
        data_path = os.path.join(root_dir_path, 'Model_Stealing_Attack/MSA_MNIST_stolen_model.pth')
    except FileNotFoundError:
        print('The stolen model should be trained with the --train argument')
    attack_model.load_state_dict(torch.load(data_path))
    attack_model.eval()

    data_path = os.path.join(root_dir_path, 'Training_data/data_MNIST')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    mnist_trainset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    mnist_testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    combined_dataset = torch.utils.data.ConcatDataset([mnist_trainset, mnist_testset])
    trainloader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=1,
        shuffle=True)

    def diff(target, stolen):
        return torch.abs(target.sub(stolen)).sum().item()

    # Evaluate the stolen model
    print(f'calculating test error on the MNIST database')
    cumulated_error = 0
    total = 0
    similar = 0
    different = 0
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs_target = target_model(inputs)
            outputs_stolen = attack_model(inputs)
            cumulated_error += diff(outputs_target, outputs_stolen)
            total += 1
            _, index_output_target = torch.max(outputs_target, 1)
            _, index_outputs_stolen = torch.max(outputs_stolen, 1)
            if index_output_target == index_outputs_stolen:
                similar += 1
            else:
                different += 1


    print(f'calculating uniform error on 10000 random inputs')
    cumulated_difference = 0
    with torch.no_grad():
        for i in range(10000):
            inputs = torch.rand(1, 28*28)
            outputs_target = target_model(inputs)
            outputs_stolen = attack_model(inputs)
            loss = diff(outputs_target, outputs_stolen)
            cumulated_difference += loss
    print(f'the test error between the models is: {cumulated_error/total}')
    print(f'the uniform error between the models is: {cumulated_difference / 10000}')
    print(f'Both models answer the same in {similar} cases, and different in {different} cases,'
          f' they agree on {similar/(similar+different)}% of the data')


if __name__ == '__main__':
    run()