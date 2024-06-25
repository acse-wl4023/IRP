import os
import numpy as np
import torch
from torch.utils.data import Dataset

def extract_timestep(filename):
    base = os.path.splitext(filename)[0]  # 移除.npy后缀
    timestep = base.split('_')[-1]  # 提取最后的时间步数值
    return float(timestep)


# get the data in one folder, and return a np.array
def read_data(_files):
    all_data = []
    for file in _files:
        _data = np.load(file)
        all_data.append(np.expand_dims(_data, axis=0))
    if all_data:
        merged_array = np.concatenate(all_data, axis=0)
        return merged_array
    else:
        merged_array = None
        return merged_array


def get_all_nodes(folders):
    all_data = []
    len_of_each_case = []
    length = 0
    len_of_each_case.append(length)
    for folder in folders:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if
                 f.startswith('foward_soln_timestep_') and f.endswith('.npy')]
        data = read_data(files)
        length = length + len(files)
        len_of_each_case.append(length)
        all_data.append(data)
    return np.concatenate(all_data, axis=0), len_of_each_case


def segment_data(data, len_of_each_case, window_size=10, step_size=10):
    all_data = []
    for i in range(len(len_of_each_case)-1):
        for j in range(len_of_each_case[i], len_of_each_case[i+1]-window_size+1, step_size):
            all_data.append(np.expand_dims(data[j:j+window_size], axis=0))
        # print("case: ", i, ", length: ", len(all_data))
        if (len_of_each_case[i+1]-window_size-len_of_each_case[i]) % step_size != 0:
            all_data.append(np.expand_dims(data[len_of_each_case[i+1]-window_size:len_of_each_case[i+1]], axis=0))
    return np.concatenate(all_data, axis=0)


class Preprocessor:
    def __init__(self):
        self.max = None
        self.min = None

    def fit(self, data):
        self.max = np.max(data, axis=(0, 1, 2))
        self.min = np.min(data, axis=(0, 1, 2))

    def transform(self, x):
        transformed_data = x/self.max

        return transformed_data

    def inverse_transform(self, x):
        recon_data = x*self.max

        return recon_data
    
class MyDataset(Dataset):
    def __init__(self, x_nodes, edge_index, edge_weight, y_nodes):
        self.x_nodes = x_nodes
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.y_nodes = y_nodes

    def __len__(self):
        return self.x_nodes.shape[0]

    def __getitem__(self, idx):
        return self.x_nodes[idx], self.edge_index, self.edge_weight, self.y_nodes[idx]