import torch
from torch.utils.data import random_split
import Models.Get_data as Gd
import numpy as np
import os
import scipy

def load_data(window_size, step_size, path_of_data):
    directory = path_of_data
    folders = [os.path.join(directory, f, 'hessian_') for f in os.listdir(directory) if f.startswith('case_')]

    pos = torch.tensor(np.load(directory+"/case_0/hessian_/xy_coords.npy"), dtype=torch.float32)

    dataset, length = Gd.get_all_nodes(folders[:-1])
    dataset = np.expand_dims(dataset[:, :, 0], axis=2)

    dataset = torch.tensor(dataset, dtype=torch.float32)
    dataset = Gd.segment_data_torch(dataset, length, window_size, step_size)

    data = Gd.MyDataset(dataset[:, 0:5, :, :], dataset[:, -5:, :, :])
    train_set, val_set = random_split(data, [0.8, 0.2])

    dataset_1, length = Gd.get_all_nodes(folders[-1:])
    dataset_1 = np.expand_dims(dataset_1[:, :, 0], axis=2)

    dataset_1 = torch.tensor(dataset_1, dtype=torch.float32)
    dataset_1 = Gd.segment_data_torch(dataset_1, length, window_size, step_size)

    test_set = Gd.MyDataset(dataset_1[:, 0:5, :, :], dataset_1[:, -5:, :, :])

    return pos, dataset, dataset_1, train_set, val_set, test_set