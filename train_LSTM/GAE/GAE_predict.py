import torch
from load_data import load_data
from train import RMSELoss, NRMSE, train
from torch.utils.data import random_split
import Models.Get_data as Gd
import numpy as np
from Models.GAE_LSTM import GAE_LSTM_seq2seq
from torchinfo import summary
import pickle
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import scipy
import yaml
import random
from test import test

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(66)  # 你可以选择任意一个数字作为种子
    # print(11111)
    # 检查是否有 GPU 可用
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
    with open("/home/wl4023/github_repos/IRP/train_LSTM/GAE/setting.yaml", "r") as file:
        config = yaml.safe_load(file)

    latent_space = config['experiment']['latent_space']
    window_size = config['experiment']['window_size']
    step_size = config['experiment']['step_size']

    data_directory = config['paths']['data_directory']
    save_path = config['paths']['save_path']
    save_path = save_path+f"Latent_space_{latent_space}/"
    os.makedirs(save_path, exist_ok=True)

    clusters = torch.load(config['paths']['cluster_file']+f'/Latent space {latent_space}/clusters.pt')
    centroids = torch.load(config['paths']['cluster_file']+f'Latent space {latent_space}/centroids.pt')

    with open(config['paths']['preprocessor_file'], 'rb') as f:
        preprocessor = pickle.load(f)


    # load data
    edge_index, edge_weight, pos, dataset, dataset_1, train_set, val_set, test_set = load_data(window_size, step_size, data_directory)

    # load model
    model = GAE_LSTM_seq2seq(latent_space=latent_space,
                             hidden_size=config['model']['hidden_size'],
                             input_node_channel=config['model']['input_node_channel'],
                             output_node_channel=config['model']['output_node_channel'],
                             num_mp_layers=config['model']['num_mp_layers'],
                             clusters=clusters,
                             centroids=centroids,
                             hidden_channels=config['model']['hidden_channels'],
                             n_mlp_mp=config['model']['n_mlp_mp'])
    summary(model, input_data=(preprocessor.transform(train_set[0][0]), 5, edge_index, edge_weight, pos))

    # train
    model = model.to(device)
    edge_index, edge_weight, pos = edge_index.to(device), edge_weight.to(device), pos.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = RMSELoss()
    metric = NRMSE(dataset)
    metric_test = NRMSE(dataset_1)
    # path_to_save = save_path+f'/GAE_LSTM.pth'
    # print(save_path+f'/GAE_LSTM.pth')

    train_RMSELoss_list, val_RMSELoss_list, train_NRMSELoss_list, val_NRMSELoss_list, test_RMSELoss_list, test_NRMSELoss_list = train(model,
                                                                                                                                    train_set,
                                                                                                                                    val_set,
                                                                                                                                    test_set,
                                                                                                                                    edge_index,
                                                                                                                                    edge_weight,
                                                                                                                                    criterion=criterion,
                                                                                                                                    pos=pos,
                                                                                                                                    metric=metric,
                                                                                                                                    metric_test=metric_test,
                                                                                                                                    optimizer=optimizer,
                                                                                                                                    preprocessor=preprocessor,
                                                                                                                                    device=device,
                                                                                                                                    epochs=config['train']['epoch'],
                                                                                                                                    seq_len=config['train']['seq_len'],
                                                                                                                                    path_to_save=save_path+f'GAE_LSTM.pth')
    # save train loss fig
    fig1, axs = plt.subplots(1, 2, figsize=(12, 6))
    # train_RMSELoss_list, val_RMSELoss_list, train_NRMSELoss_list, val_NRMSELoss_list
    axs[0].plot(train_RMSELoss_list, label='train RMSE Loss')
    axs[0].plot(val_RMSELoss_list, label='val RMSE Loss')
    axs[0].plot(test_RMSELoss_list, label='test RMSE Loss')
    axs[0].set_title("RMSE Loss")
    axs[0].legend()

    axs[1].plot(train_NRMSELoss_list, label='train NRMSE Loss')
    axs[1].plot(val_NRMSELoss_list, label='val NRMSE Loss')
    axs[1].plot(test_NRMSELoss_list, label='test NRMSE Loss')
    axs[1].set_title("NRMSE Loss")
    axs[1].legend()

    plt.tight_layout()  # 调整布局以防止重叠
    # plt.savefig('CAE_train_loss.png')  # 你可以更改文件名或格式
    plt.savefig(save_path+f'GAE_LSTM_train_loss.png')

    test(model, test_set, edge_index, edge_weight, 5, pos, criterion, metric_test, preprocessor, device, save_path)
