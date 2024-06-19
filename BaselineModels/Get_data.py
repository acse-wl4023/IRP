import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch


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


def get_all_data(folders):
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


class Preprocessor:
    def __init__(self, _data, n_components=80):
        self.data = _data
        self.n_components = n_components
        self.pca = None
        self.scaler = None

    def fit(self):
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(self.data)
        x = self.pca.transform(self.data)
        self.scaler = StandardScaler()
        self.scaler.fit(x)

    def transform(self, data):
        compressed_data = self.pca.transform(data)
        scaled_data = self.scaler.transform(compressed_data)
        print(scaled_data.max())
        return scaled_data

    def inverse_transform(self, scaled_data):
        compressed_data = self.scaler.inverse_transform(scaled_data)
        recon_data = self.pca.inverse_transform(compressed_data)
        return recon_data


def data_panelling(data, len_of_each_case, window_size=10, step_size=10):
    all_data = []
    for i in range(len(len_of_each_case)-1):

        for j in range(len_of_each_case[i], len_of_each_case[i+1]-window_size+1, step_size):
            all_data.append(np.expand_dims(data[j:j+window_size, :], axis=0))
        # print("case: ", i, ", length: ", len(all_data))
        if (len_of_each_case[i+1]-window_size-len_of_each_case[i]) % step_size != 0:
            all_data.append(np.expand_dims(data[len_of_each_case[i+1]-window_size:len_of_each_case[i+1], :], axis=0))
    return np.concatenate(all_data, axis=0)


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PreprocessorCNN:
    def __init__(self, data, n_channel):
        self.data = data
        self.n_channel = n_channel
        self.scalerList = []

    def fit(self):
        for i in range(self.n_channel):
            scaler = StandardScaler()
            scaler.fit(self.data[:, :, i])
            self.scalerList.append(scaler)

    def transform(self, x):
        transformed_data = np.zeros_like(x)
        for i in range(self.n_channel):
            transformed_data[:, :, i] = self.scalerList[i].transform(x[:, :, i])

        return transformed_data

    def inverse_transform(self, x):
        recon_data = np.zeros_like(x)
        for i in range(self.n_channel - 1, -1, -1):
            recon_data[:, :, i] = self.scalerList[i].inverse_transform(x[:, :, i])

        return recon_data
