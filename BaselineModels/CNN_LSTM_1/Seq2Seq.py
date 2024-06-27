import torch.nn as nn
import torch


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
    def __init__(self, hidden_size, output_size, output_seq_len):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_seq_len = output_seq_len
        
        self.sigmoid = nn.Sigmoid()
        
        self.W_xi = nn.Linear(self.hidden_size, self.output_size, bias=True)
        self.W_hi = nn.Linear(self.output_size, self.output_size, bias=True)
        
        self.W_xf = nn.Linear(self.hidden_size, self.output_size, bias=True)
        self.W_hf = nn.Linear(self.output_size, self.output_size, bias=True)
        
        self.W_xo = nn.Linear(self.hidden_size, self.output_size, bias=True)
        self.W_ho = nn.Linear(self.output_size, self.output_size, bias=True)
        
        self.W_xc = nn.Linear(self.hidden_size, self.output_size, bias=True)
        self.W_hc = nn.Linear(self.output_size, self.output_size, bias=True)

        self.ini_h = nn.Linear(self.hidden_size, self.output_size)
        self.ini_c = nn.Linear(self.hidden_size, self.output_size)


    
    def forward(self, x, h, c):
        
        h_t = self.ini_h(h)
        c_t = self.ini_c(c)

        output = []
        for t in range(self.output_seq_len):
            
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
    def __init__(self, input_size, hidden_size, output_size, output_seq_len):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_seq_len = output_seq_len
        
        self.encoder = Encoder(self.input_size, self.hidden_size)
        self.decoder = Decoder(self.hidden_size, self.output_size, self.output_seq_len)
        
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        h_t, c_t = self.encoder(x)

        output = self.decoder(h_t, h_t, c_t)
        
        output = self.leakyrelu(output)
        
        return output
        