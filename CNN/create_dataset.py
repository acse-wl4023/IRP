import Get_data as Gd
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def create_train_val_set(folders, train_batchsize=5, val_batchsize=5):
    dataset, _ = Gd.get_all_nodes(folders)

    train_np, val_np = train_test_split(dataset, test_size=0.2, shuffle=True)
    train_np = np.expand_dims(train_np[:, :, 0], axis=2)
    val_np = np.expand_dims(val_np[:, :, 0], axis=2)

    train_tensor = torch.tensor(train_np, dtype=torch.float).permute(0, 2, 1)
    train_tensor_set = Gd.MyDataset(train_tensor)

    val_tensor = torch.tensor(val_np, dtype=torch.float).permute(0, 2, 1)
    val_tensor_set = Gd.MyDataset(val_tensor)

    train_loader = DataLoader(train_tensor_set, batch_size=train_batchsize, shuffle=True)
    val_loader = DataLoader(val_tensor_set, batch_size=val_batchsize, shuffle=True)

    return train_loader, val_loader, train_tensor, val_tensor