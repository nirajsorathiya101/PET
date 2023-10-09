import csv
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import NeuralNetworks
from torch.utils.data.sampler import SubsetRandomSampler
import argparse



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train both target and shadow models')
    opt = parser.parse_args()
    trainModels = opt.train

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Absolute paths
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(main_dir_path, 'Training_data/data')

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    # Dividing training dataset in 4 equal parts
    dataset_size = len(trainset)
    indices = list(range(dataset_size))
    split = len(trainset) // 4
    np.random.shuffle(indices)
    train_target, train_shadow = SubsetRandomSampler(indices[:split]), SubsetRandomSampler(indices[split+1:split*2])

    test_target, test_shadow = SubsetRandomSampler(indices[split*2+1:split*3]), SubsetRandomSampler(indices[split*3+1:split*4])

    trainloader_target = torch.utils.data.DataLoader(trainset, batch_size=20,
                                                     sampler=train_target)
    trainloader_shadow = torch.utils.data.DataLoader(trainset, batch_size=20,
                                                     sampler=train_shadow)
    testloader_shadow = torch.utils.data.DataLoader(trainset, batch_size=1,
                                                    sampler=test_shadow)
    testloader_target = torch.utils.data.DataLoader(trainset, batch_size=1,
                                                    sampler=test_target)

    # dynamic path generation
    model_path_target = os.path.join(dir_path, 'MIA_target_model_cifar.pth')
    model_path_shadow = os.path.join(dir_path, 'MIA_shadow_model_cifar.pth')
    data_path_shadow_train = os.path.join(main_dir_path, 'Training_data/MIA_training_data.csv')
    data_path_shadow_test = os.path.join(main_dir_path, 'Training_data/MIA_test_data.csv')

    # dictionary of models with features for easy referencing
    models = {'target_model': {'path': model_path_target, 'trainloader': trainloader_target,
                               'testloader': testloader_shadow, 'csv': data_path_shadow_test,
                               'trainsampler':train_target, 'testsampler':  test_target},
              'shadow_model': {'path': model_path_shadow, 'trainloader': trainloader_shadow,
                               'testloader': testloader_target, 'csv': data_path_shadow_train,
                               'trainsampler':train_shadow, 'testsampler': test_shadow}}

    # Training the  target and shadow models
    if trainModels:
        print('training target and shadow models')
        for model, features in models.items():
            net = NeuralNetworks.CNNNet()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            EPOCHS = 200
            print(f'training {model} for {EPOCHS} epochs')
            with tqdm(total=EPOCHS) as pbar:
                for epoch in range(EPOCHS):  # loop over the dataset multiple times

                    for i, data in enumerate(features['trainloader'], 0):
                        inputs, labels = data
                        optimizer.zero_grad()
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    pbar.update(1)
                torch.save(net.state_dict(), features['path'])
                print('Finished Training ' + model + ', state dictionary saved under ' + features['path'])
                pbar.reset()

    # crating our training and test data for the attack model
    for model, features in models.items():
        net = NeuralNetworks.CNNNet()
        try:
            net.load_state_dict(torch.load(features['path']))
        except FileNotFoundError:
            print('you should use the --train argument to train the models')

        features['trainloader'] = torch.utils.data.DataLoader(trainset, batch_size=1, sampler=features['trainsampler'])  # Change batch size to 1
        path = os.path.join(dir_path, features['csv'])
        loaders = {'1': features['trainloader'], '0': features['testloader']}
        with open(path, mode='w') as data_file:
            with torch.no_grad():
                for truth, loader in loaders.items():
                    for i, data in tqdm(enumerate(loader, 0)):
                        inputs, labels = data
                        n = torch.nn.Softmax(dim=1)
                        output = n(net(inputs))  # normalize output, remove gradient tracking
                        value = output[0].numpy()  # change tensor to numpy array
                        value.sort(axis=0)
                        value = value[7:10]  # select the 3 biggest value in array
                        reference = torch.Tensor([1] if truth == '1' else [0])
                        cvs_format = np.concatenate((value, reference.numpy()))  # format to fit on one line
                        data_writer = csv.writer(data_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                        data_writer.writerow(cvs_format)
                print('Attack model training dataset from ' + model + ' saved at ' + features['csv'])

    # Testing the accuracy of the target model
    for model, features in models.items():
        sum = 0
        net = NeuralNetworks.CNNNet()
        net.load_state_dict(torch.load(features['path']))
        loader = features['testloader']
        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                if predicted.item() == labels.item():
                    sum += 1
            print('accuracy from {} = {} on testing dataset'.format(model, sum/split*100))

    for model, features in models.items():
        sum = 0
        net = NeuralNetworks.CNNNet()
        net.load_state_dict(torch.load(features['path']))
        loader = features['trainloader']
        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                if predicted.item() == labels.item():
                    sum += 1
            print('accuracy from {} = {} on training dataset'.format(model, sum/split*100))


if __name__ == '__main__':
    run()