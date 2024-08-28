import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def draw_pic(x, y, predict, coords, path_to_save):
    fig, axs = plt.subplots(3, 5, figsize=(30, 18))
    for i in range(axs.shape[-1]):
        axs[0][i].scatter(coords[:,0],coords[:,1],s = 5, c=x[i], cmap='bwr')
        axs[0][i].set_title(f'Origin Input {i+1}')

    for i in range(axs.shape[-1]):
        axs[1][i].scatter(coords[:,0],coords[:,1],s = 5, c=y[i], cmap='bwr')
        axs[1][i].set_title(f'Real Output {i+1}')

    for i in range(axs.shape[-1]):
        axs[2][i].scatter(coords[:,0],coords[:,1],s = 5, c=predict[i], cmap='bwr')
        axs[2][i].set_title(f'Real Output {i+1}')

    plt.savefig(path_to_save+'/GConvLSTM_output.png')

def test(model, test_set, edge_index, edge_weight, seq_len, pos, criterion, metric, preprocessor, device, path_to_save):
    test_RMSELoss_list = []
    test_NRMSELoss_list = []

    recon_predict_list = []
    with torch.no_grad():
        test_loss = 0
        test_metric = 0
        for x, y in tqdm(test_set):
            input = preprocessor.transform(x).to(device)
            predict = model(input, seq_len, edge_index, edge_weight, pos)

            recon_predict = preprocessor.inverse_transform(predict.cpu())

            # recon_predict_list.append(recon_predict)
            test_RMSELoss_list.append(criterion(y, recon_predict).item())
            test_NRMSELoss_list.append(metric(y, recon_predict).item())
            recon_predict_list.append(recon_predict)

    recon_predict_list = torch.stack(recon_predict_list, dim=0)
    test_rmse = np.array(test_RMSELoss_list)
    test_nrmse = np.array(test_NRMSELoss_list)

    test_loss = test_rmse.mean()
    test_metric = test_nrmse.mean()

    print(f'Ave test loss: {test_loss}, Ave test metric: {test_metric}')

    txt_file = path_to_save+'/test_results.txt'

    # 将数据写入文本文件
    with open(txt_file, mode='w') as file:
        file.write(f'Ave test loss: {test_loss}\n')
        file.write(f'Ave test metric: {test_metric}\n')

    x = np.arange(0, len(test_set))

    fig1, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(x, test_rmse, label='test RMSE Loss')
    axs[0].set_title("RMSE Loss")
    axs[0].legend()

    axs[1].scatter(x, test_nrmse, label='test NRMSE Loss')
    axs[1].set_title("NRMSE Loss")
    axs[1].legend()


    plt.tight_layout()  # 调整布局以防止重叠
    # plt.savefig('CAE_test_loss.png')  # 你可以更改文件名或格式
    plt.savefig(path_to_save+'/GConvLSTM_test_loss.png')

    x, y = test_set[0]
    predict = recon_predict_list[0]
    draw_pic(x.cpu().numpy(), y.cpu().numpy(), predict.cpu().numpy(), pos.cpu().numpy(), path_to_save)


