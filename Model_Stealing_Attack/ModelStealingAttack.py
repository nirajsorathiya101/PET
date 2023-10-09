import torch
import os
import sys
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
from NeuralNetworks import NeuralNetworks


class ModelStealingAttack:

    def dataset_gen(self, size):
        dataset = []
        for i in range(size):
            dataset.append((torch.rand(1, 28*28), 0))  # 0 is a filler value for the dataloader
        return dataset

    def MSAAttack(self):

        device = torch.device('cpu')  # default is cpu
        # this checks if a cuda gpu is available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('yes gpu')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Load the target model

        target_model = NeuralNetworks.NetMNISTMSA().to(device)
        root_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        try:
            data_path = os.path.join(root_dir_path, 'NeuralNetworks/MSA_MNIST_net.pth')
        except FileNotFoundError:
            print('You need to train the target model first with MSA_MNIST_generator.py')
        target_model.load_state_dict(torch.load(data_path))
        target_model.eval()

        dataset_size = 5000
        dataset = self.dataset_gen(dataset_size)  # dataset of 5000 tensors to be put in batches

        # The dataloader allows us to train in batches
        trainloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=False)

        stolen_model = NeuralNetworks.NetMNISTMSA().to(device)  # Model that will host a copy of the target model
        # The learning rate has a big influence on the results,
        optimizer = optim.SGD(stolen_model.parameters(), lr=0.003, momentum=0.9)

        criterion = nn.L1Loss()  # Simple mean absolute error to compare outputs
        m = torch.nn.Softmax(dim=1)  # Used to format the model outputs for the loss function
        EPOCHS = 1000  # like in the paper
        print(f'copying the target model using {dataset_size} randomly generated samples for {EPOCHS} epochs')
        with tqdm(total=EPOCHS) as pbar:
            for epoch in range(EPOCHS):
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)  # transfers the tensors on gpu
                    optimizer.zero_grad()
                    outputs_target = target_model(inputs)
                    outputs_stolen = stolen_model(inputs)
                    loss = criterion(m(outputs_stolen), m(outputs_target))
                    loss.backward()
                    optimizer.step()
                pbar.update(1)

        # save the stolen model
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(dir_path, 'MSA_MNIST_stolen_model.pth')
        torch.save(stolen_model.state_dict(), data_path)
        print('Finished training the model stealing model, it is saved under MSA_MNIST_stolen_model.pth')