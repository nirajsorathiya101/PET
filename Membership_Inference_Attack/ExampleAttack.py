import os
import torch
import torch.utils.data as utils
from tqdm import tqdm
import DataLoader
import InferenceAttack
import argparse


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the attack model')
    opt = parser.parse_args()
    trainModels = opt.train

    device = torch.device('cpu')  # default is cpu
    # this checks if a cuda gpu is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('yes gpu')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    attack = InferenceAttack.MembershipInferenceAttack(train=trainModels)
    model = attack.net
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        data_path = os.path.join(dir_path, 'MIA_attack_model.pth')
    except FileNotFoundError:
        print('you should use the --train argument to train the attack model')
    model.load_state_dict(torch.load(data_path))
    model.eval()

    dataset = DataLoader.MiaDataLoader('Training_data/MIA_test_data.csv')
    trainloader = utils.DataLoader(dataset=dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=2)
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    print('Calculating the accuracy of the attack model:')
    for i, data in tqdm(enumerate(trainloader, 0)):
        with torch.no_grad():
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # transfers the tensors on gpu
            outputs = model(inputs)
            _, index_output = torch.max(outputs, 1)
            if index_output == 1 and index_output.eq(labels.item()):
                true_positive += 1
            elif index_output == 0 and index_output.eq(labels.item()):
                true_negative += 1
            elif index_output == 0:
                false_negative += 1
            elif index_output == 1:
                false_positive += 1
    print(f'true positive: {true_positive}   false positive: {false_positive}')
    print(f'true negative: {true_negative}   false negative: {false_negative}')
    print(f' precision: {true_positive/(true_positive+false_positive)}      recall: {true_positive/(true_positive+false_negative)}')


if __name__ == '__main__':
    run()
