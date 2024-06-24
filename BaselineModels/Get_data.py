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
    def __init__(self):
        self.mean_1 = None
        self.std_1 = None

        self.mean_2 = None
        self.std_2 = None

        self.pca = None

    def fit(self, data, n_components=50):
        self.mean_1 = np.mean(data, axis=(0, 1))
        self.std_1 = np.std(data, axis=(0, 1))
        normalized_data = (data-self.mean_1)/self.std_1

        shape = data.shape
        reshape_normalized_data = normalized_data.reshape(shape[0]*shape[1], shape[2])
        # reshape_normalized_data = data.reshape(shape[0]*shape[1], shape[2])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(reshape_normalized_data)
        pca_data = self.pca.transform(reshape_normalized_data)

        reshape_pca_data = pca_data.reshape(shape[0], shape[1], n_components)
        self.mean_2 = np.mean(reshape_pca_data, axis=(0, 1))
        self.std_2 = np.std(reshape_pca_data, axis=(0, 1))


    def transform(self, data):
        normalized_data = (data-self.mean_1)/self.std_1

        shape = data.shape
        reshape_normalized_data = normalized_data.reshape(shape[0]*shape[1], shape[2])
        # reshape_normalized_data = data.reshape(shape[0]*shape[1], shape[2])
        pca_data = self.pca.transform(reshape_normalized_data)

        reshape_pca_data = pca_data.reshape(shape[0], shape[1], self.pca.n_components_)
        normalized_pca_data = (reshape_pca_data-self.mean_2)/self.std_2

        return normalized_pca_data


    def inverse_transform(self, normalized_pca_data):
        reshape_pca_data = normalized_pca_data*self.std_2+self.mean_2 #(num_input, seq_len, num_feature)

        shape = reshape_pca_data.shape
        pca_data = reshape_pca_data.reshape(shape[0]*shape[1], shape[2]) #(num_input*seq_len, num_feature)
        reshape_normalized_data = self.pca.inverse_transform(pca_data) #(num_input*seq_len, original_num_feature)

        shape_ = reshape_normalized_data.shape[-1]
        # recon_data = reshape_normalized_data.reshape(shape[0], shape[1], shape_) #(num_input, seq_len, original_num_feature)
        normalized_data = reshape_normalized_data.reshape(shape[0], shape[1], shape_) #(num_input, seq_len, original_num_feature)
        recon_data = normalized_data*self.std_1+self.mean_1
        
        return recon_data


def segment_data(data, len_of_each_case, window_size=10, step_size=10):
    all_data = []
    for i in range(len(len_of_each_case)-1):

        for j in range(len_of_each_case[i], len_of_each_case[i+1]-window_size+1, step_size):
            all_data.append(np.expand_dims(data[j:j+window_size], axis=0))
        # print("case: ", i, ", length: ", len(all_data))
        if (len_of_each_case[i+1]-window_size-len_of_each_case[i]) % step_size != 0:
            all_data.append(np.expand_dims(data[len_of_each_case[i+1]-window_size:len_of_each_case[i+1]], axis=0))
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
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=(0, 1))
        self.std = np.std(data, axis=(0, 1))

        # if np.any(self.std == 0):
        #     print(1111)

    def transform(self, x):
        transformed_data = (x-self.mean)/self.std

        return transformed_data

    def inverse_transform(self, x):
        recon_data = x*self.std+self.mean

        return recon_data
