import torch
import torch.nn as nn
from tqdm import tqdm

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_true, y_pred):
        return torch.sqrt(self.mse(y_true, y_pred))
    
class NRMSE(nn.Module):
    def __init__(self, total_dataset):
        super(NRMSE, self).__init__()
        self.rmse = RMSELoss()
        self.factor = total_dataset.max()-total_dataset.min()

    def forward(self, y_true, y_pred):
        return self.rmse(y_true, y_pred)/self.factor
    
def train(model, train_set, val_set, test_set, edge_index, edge_weight, criterion, pos, metric, metric_test, optimizer, preprocessor, device, scheduler=None, epochs=30, seq_len=5, path_to_save=None):
    train_NRMSELoss_list = []
    train_RMSELoss_list = []

    val_NRMSELoss_list = []
    val_RMSELoss_list = []

    test_RMSELoss_list = []
    test_NRMSELoss_list = []

    min_loss = 10000
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_metric = 0
        for x, y in tqdm(train_set):
            input = preprocessor.transform(x).to(device)
            label = preprocessor.transform(y).to(device)
            # x, y = x.to(device), y.to(device)
            # x = x.to(device)
            optimizer.zero_grad()
            predict = model(input, seq_len, edge_index, edge_weight, pos)
            loss = criterion(label, predict)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()

            recon_predict = preprocessor.inverse_transform(predict.cpu())
            # print(recon_predict.max(), train_loss, len(train_loader))
            train_loss += criterion(y, recon_predict).item()
            train_metric += metric(y, recon_predict).item()

            # train_loss += criterion(y.cpu(), predict.cpu()).item()
            # train_metric += metric(y.cpu(), predict.cpu()).item()
        if scheduler != None:
            scheduler.step()

        train_loss /= len(train_set)
        # print(train_loss, len(train_loader))
        train_metric /= len(train_set)
        train_RMSELoss_list.append(train_loss)
        train_NRMSELoss_list.append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_metric = 0
            for x, y in val_set:
                input = preprocessor.transform(x).to(device)
                # x = x.to(device)
                predict = model(input, seq_len, edge_index, edge_weight, pos)

                recon_predict = preprocessor.inverse_transform(predict.cpu())
                val_loss += criterion(y, recon_predict).item()
                val_metric += metric(y, recon_predict).item()
                # val_loss += criterion(y, predict.cpu()).item()
                # val_metric += metric(y, predict.cpu()).item()
    
            val_loss /= len(val_set)
            val_metric /= len(val_set)
            val_RMSELoss_list.append(val_loss)
            val_NRMSELoss_list.append(val_metric)

        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_metric = 0
            for x, y in test_set:
                input = preprocessor.transform(x).to(device)
                predict = model(input, 5, edge_index, edge_weight, pos)

                recon_predict = preprocessor.inverse_transform(predict.cpu())

                # recon_predict_list.append(recon_predict)
                test_loss += criterion(y, recon_predict).item()
                test_metric += metric_test(y, recon_predict).item()
            
            test_loss /= len(test_set)
            test_metric /= len(test_set)
            test_RMSELoss_list.append(test_loss)
            test_NRMSELoss_list.append(test_metric)

        # recon_predict_list = torch.stack(recon_predict_list, dim=0).numpy()

        # print(f'Ave test loss: {test_loss}, Ave test metric: {test_metric}')

    

        print(f'Epoch {epoch + 1}/{epochs}, train Loss: {train_loss}, NRMSE_train_loss: {train_metric}, val Loss: {val_loss}, NRMSE_val_loss: {val_metric}, Ave test loss: {test_loss}, Ave test metric: {test_metric}')
        
        if min_loss >= test_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), path_to_save)

    return train_RMSELoss_list, val_RMSELoss_list, train_NRMSELoss_list, val_NRMSELoss_list, test_RMSELoss_list, test_NRMSELoss_list