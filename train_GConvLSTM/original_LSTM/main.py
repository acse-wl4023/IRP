import torch
from load_data import load_data
from train import RMSELoss, NRMSE, train
from torch.utils.data import random_split
import Models.Get_data as Gd
import numpy as np
from Models.LSTM import LSTM_seq2seq
from torchinfo import summary
import pickle
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import scipy
import yaml
from test import test
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(66)  # 你可以选择任意一个数字作为种子
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    # 检查 GPU 的数量
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # 获取每个 GPU 的名称
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # 设置默认使用的 GPU 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # get settings
    with open("/home/wl4023/github_repos/IRP/train_GConvLSTM/original_LSTM/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    save_path = config['paths']['save_path']
    save_path = save_path+f"LSTM/"
    os.makedirs(save_path, exist_ok=True)
    
    pos, dataset, dataset_1, train_set, val_set, test_set = load_data(config["experiment"]["window_size"],
                                                                      config["experiment"]["step_size"],
                                                                      config["paths"]["data_directory"])
    with open(config['paths']['preprocessor_file'], 'rb') as f:
        preprocessor = pickle.load(f)
    
    # load model
    model = LSTM_seq2seq(97149, 100)
    summary(model, input_data=(train_set[0][0], 5))

    # #train model
    # model = model.to(device)
    # edge_index, edge_weight, pos = edge_index.to(device), edge_weight.to(device), pos.to(device)
    # optimizer = optim.Adam(model.parameters())
    # criterion = RMSELoss()
    # metric = NRMSE(dataset)
    # metric_test = NRMSE(dataset_1)

    # print(config['train']['epoch'])

    # train_RMSELoss_list, val_RMSELoss_list, train_NRMSELoss_list, val_NRMSELoss_list, test_RMSELoss_list, test_NRMSELoss_list = train(model=model,
    #                                                                                                                                   train_set=train_set,
    #                                                                                                                                   val_set=val_set,
    #                                                                                                                                   test_set=test_set,
    #                                                                                                                                   edge_index=edge_index,
    #                                                                                                                                   edge_weight=edge_weight,
    #                                                                                                                                   criterion=criterion,
    #                                                                                                                                   metric=metric,
    #                                                                                                                                   metric_test=metric_test,
    #                                                                                                                                   optimizer=optimizer,
    #                                                                                                                                   preprocessor=preprocessor,
    #                                                                                                                                   device=device,
    #                                                                                                                                   epochs=config['train']['epoch'],
    #                                                                                                                                   seq_len=config['train']['seq_len'],
    #                                                                                                                                   path_to_save=save_path+f'GConvLSTM.pth')

    # test(model, test_set, edge_index, edge_weight, 5, pos, criterion, metric_test, preprocessor, device, save_path)