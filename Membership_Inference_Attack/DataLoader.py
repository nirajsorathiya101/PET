import os
import torch
import numpy as np
import torch.utils.data as utils

# This is used to load the generated .csv files
class MiaDataLoader(utils.Dataset):

    def __init__(self, path):
        main_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        data_path = os.path.join(main_dir_path, path)
        database = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
        self.len = database.shape[0]

        self.feed_data = torch.from_numpy(database[:, [0, 1, 2]])
        self.groundTruth_data = torch.from_numpy(database[:, [-1]])

    def __getitem__(self, index):
        return self.feed_data[index], self.groundTruth_data[index]

    def __len__(self):
        return self.len
