import torch.nn as nn
import torch
import torch.utils.data as utils
import torch.optim as optim
from tqdm import tqdm
import DataLoader
import sys
import os
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)
from NeuralNetworks import NeuralNetworks
class MembershipInferenceAttack:

    def __init__(self, train=False):
        if train:
            self.model = self.mianet()
        self.net = NeuralNetworks.NetMIAAttack()

    def mianet(self):
        device = torch.device('cpu') # default is cpu
        # this checks if a cuda gpu is available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('yes gpu')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # create the dataloader from a csv file
        dataset = DataLoader.MiaDataLoader('Training_data/MIA_training_data.csv')

        trainloader = utils.DataLoader(dataset=dataset,
                                        batch_size=20,
                                        shuffle=True,
                                        num_workers=2)

        net = NeuralNetworks.NetMIAAttack().to(device) # transfers the model on gpu

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)
        EPOCHS = 100
        print(f'training attack model for {EPOCHS} epochs')
        with tqdm(total=EPOCHS) as pbar:
            for epoch in range(EPOCHS):
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device) # transfers the tensors on gpu
                    labels = labels.view(-1).long()
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                pbar.update(1)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(dir_path, 'MIA_attack_model.pth')
        torch.save(net.state_dict(), data_path)
        print('Finished Training MIA attack model, saved under MIA_attack_model.pth')

        return net

