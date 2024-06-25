import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.sigmoid = nn.Sigmoid()
        
        self.W_xi = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.W_xf = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.W_xo = nn.Linear(input_size, hidden_size, bias=True)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.W_xc = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hc = nn.Linear(hidden_size, hidden_size, bias=True)
        
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size, device=device), torch.zeros(batch_size, self.hidden_size, device=device)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_t, c_t = self.init_hidden(batch_size, x.device)
        for t in range(seq_len):
            x_t = x[:, t, :]
            i_t = self.sigmoid(self.W_xi(x_t)+self.W_hi(h_t))
            f_t = self.sigmoid(self.W_xf(x_t)+self.W_hf(h_t))
            o_t = self.sigmoid(self.W_xo(x_t)+self.W_ho(h_t))
            g_t = torch.tanh(self.W_xc(x_t)+self.W_hc(h_t))
            
            c_t = f_t*c_t+i_t*g_t
            h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t
            
        
class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        
        self.sigmoid = nn.Sigmoid()
        
        self.W_xi = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.W_xf = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.W_xo = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self.W_xc = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hc = nn.Linear(hidden_size, hidden_size, bias=True)
        
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size, device=device), torch.zeros(batch_size, self.hidden_size, device=device)
    
    def forward(self, x, expected_seq_len):
        batch_size = x.shape[0]
        h_t, c_t = self.init_hidden(batch_size, x.device)
        output = []
        for t in range(expected_seq_len):
            
            i_t = self.sigmoid(self.W_xi(x)+self.W_hi(h_t))
            f_t = self.sigmoid(self.W_xf(x)+self.W_hf(h_t))
            o_t = self.sigmoid(self.W_xo(x)+self.W_ho(h_t))
            g_t = torch.tanh(self.W_xc(x)+self.W_hc(h_t))
            
            c_t = f_t*c_t+i_t*g_t
            h_t = o_t * torch.tanh(c_t)
            
            output.append(h_t.unsqueeze(1))
        output = torch.cat(output, dim=1)
        return output


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim)
        self.linear = nn.Linear(hidden_dim, self.output_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, x, expected_seq_len = 5):
        h_t, _ = self.encoder(x)
        output = self.decoder(h_t, expected_seq_len)
        
        output = self.relu(self.linear(output))
        
        return output


def train(model, train_loader, optimizer, criterion, device, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x_1, y_1 = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x_1)
            loss = criterion(y_1, output)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, train Loss: {total_loss / len(train_loader)}')


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))
    

def test(model, test_loader, test_real_loader, criterion, preprocessor, device):
    model.eval()
    latent_err = 0
    recon_err = 0
    err_y = 0
    err_x = 0
    with torch.no_grad():
        for (x, y), (x_r, y_r) in zip(test_loader, test_real_loader):
            x, y, x_r, y_r = x.to(device), y.to(device), x_r.to(device), y_r.to(device)
            output = model(x)
            latent_err += criterion(y, output).item()
            
            recon_y = preprocessor.inverse_transform(y.cpu().detach().numpy())
            recon_x = preprocessor.inverse_transform(x.cpu().detach().numpy())
            recon_output = preprocessor.inverse_transform(output.cpu().detach().numpy())

            y_r_np = y_r.cpu().detach().numpy()
            x_r_np = x_r.cpu().detach().numpy()

            recon_err += mean_squared_error(y_r_np.squeeze(), recon_output.squeeze())
            err_y += mean_squared_error(y_r_np.squeeze(), recon_y.squeeze())
            err_x += mean_squared_error(x_r_np.squeeze(), recon_x.squeeze())
        
        return latent_err/len(test_loader), recon_err/len(test_real_loader), err_y/len(test_real_loader), err_x/len(test_real_loader)
            