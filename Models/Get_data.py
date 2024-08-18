import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric import data as tgd

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
        files.sort(key=extract_timestep)
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
        self.mean = None
        self.std = None

    def fit(self, data):
        # 计算数据的整体最大值和最小值
        self.mean = torch.mean(data)
        self.std = torch.std(data)

    def transform(self, x):
        # 使用整体最大值进行归一化
        transformed_data = (x - self.mean) / self.std
        return transformed_data

    def inverse_transform(self, x):
        # 使用整体最大值和最小值进行逆变换
        recon_data = x * self.std + self.mean
        return recon_data
    
class MyDataset(Dataset):
    def __init__(self, x,  y=None, case=None):
        self.x_nodes = x
        self.y_nodes = y
        self.case = case

    def __len__(self):
        return self.x_nodes.shape[0]

    def __getitem__(self, idx):
        if self.y_nodes is None:
            return self.x_nodes[idx]
        else:
            if self.case is None:
                return self.x_nodes[idx], self.y_nodes[idx]
            else:
                return self.x_nodes[idx], self.y_nodes[idx], self.case[idx]
            

class CustomDataset(tgd.Dataset):
    def __init__(self, edge_index, node_features, edge_attr, pos):
        self.edge_index = edge_index
        # self.num_samples = num_samples
        self.node_features = node_features
        self.edge_attr = edge_attr
        self.pos = pos
        # self.transform = transform
        super().__init__()

    def len(self):
        return len(self.node_features)

    def get(self, idx):
        
        x = self.node_features[idx]

        
        # Create a Data object
        data = tgd.Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, pos=self.pos)
        
        return data
        
