import torch
from load_data import load_data
from train import RMSELoss, NRMSE, train
import numpy as np
from Models.GAE_GConvLSTM import GAE_GConvLSTM_seq2seq
from torchinfo import summary
import pickle
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import os
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
    with open("/home/wl4023/github_repos/IRP/train_GConvLSTM/GAE_GConvLSTM/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    save_path = config['paths']['save_path']
    save_path = save_path+f'Latent_space {config["experiment"]["latent_space"]}/'
    os.makedirs(save_path, exist_ok=True)
    
    edge_index, edge_weight, pos, dataset, dataset_1, train_set, val_set, test_set = load_data(config["experiment"]["window_size"],
                                                                                               config["experiment"]["step_size"],
                                                                                               config["paths"]["data_directory"])
    with open(config['paths']['preprocessor_file'], 'rb') as f:
        preprocessor = pickle.load(f)

    clusters = torch.load(config['paths']['cluster_file']+f'/Latent space {config["experiment"]["latent_space"]}/clusters.pt')
    centroids = torch.load(config['paths']['cluster_file']+f'Latent space {config["experiment"]["latent_space"]}/centroids.pt')
    
    # load model
    model = GAE_GConvLSTM_seq2seq(latent_space=config["experiment"]["latent_space"],
                                  hidden_channel_lstm=config["experiment"]["hidden_channel_lstm"],
                                  input_node_channel=config["experiment"]["input_node_channel"],
                                  output_node_channel=config["experiment"]["output_node_channel"],
                                  num_mp_layers=config["experiment"]["num_mp_layers"],
                                  clusters=clusters,
                                  centroids=centroids,
                                  hidden_channels=config["experiment"]["hidden_channels"],
                                  n_mlp_mp=config["experiment"]["n_mlp_mp"],)
    summary(model, input_data=(dataset[0, :5, :, :], 5, edge_index, edge_weight, pos))

    # #train model
    model = model.to(device)
    edge_index, edge_weight, pos = edge_index.to(device), edge_weight.to(device), pos.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = RMSELoss()
    metric = NRMSE(dataset)
    metric_test = NRMSE(dataset_1)

    train_RMSELoss_list, val_RMSELoss_list, train_NRMSELoss_list, val_NRMSELoss_list, test_RMSELoss_list, test_NRMSELoss_list = train(model=model,
                                                                                                                                      train_set=train_set,
                                                                                                                                      val_set=val_set,
                                                                                                                                      test_set=test_set,
                                                                                                                                      edge_index=edge_index,
                                                                                                                                      edge_weight=edge_weight,
                                                                                                                                      pos=pos,
                                                                                                                                      criterion=criterion,
                                                                                                                                      metric=metric,
                                                                                                                                      metric_test=metric_test,
                                                                                                                                      optimizer=optimizer,
                                                                                                                                      preprocessor=preprocessor,
                                                                                                                                      device=device,
                                                                                                                                      epochs=config['train']['epoch'],
                                                                                                                                      seq_len=config['train']['seq_len'],
                                                                                                                                      path_to_save=save_path+f'GConvLSTM.pth')
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
    plt.savefig(save_path+f'GConvLSTM_train_loss.png')
    
    
    test(model, test_set, edge_index, edge_weight, 5, pos, criterion, metric_test, preprocessor, device, save_path)