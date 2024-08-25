import torch
import torch.nn as nn
from torch.nn import Parameter
from .CAE import Encoder, Decoder
from .LSTM_Cell import LSTM_cell



class CAE_LSTM_encoder(nn.Module):
    def __init__(self, latent_space, hidden_size):
        super(CAE_LSTM_encoder, self).__init__()
        self.latent_space = latent_space # output of the cae encoder, input of the lstm encoder
        self.hidden_size = hidden_size # input of the lstm encoder

        self.cell = LSTM_cell(self.latent_space, self.hidden_size)

        self.encoder = Encoder(1, self.latent_space)

    def _init_states(self, x):
        h_t = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        c_t = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        return h_t, c_t
    
    def forward(self, x, init_states=None): # input = (batch_size, seq_length, channels, input_size):
        # 状态初始化
        if init_states is None:
            h_t, c_t = self._init_states(x)
        else:
            h_t, c_t = init_states

        seq_len = x.size(1)

        for i in range(seq_len):
            hidden_input = self.encoder(x[:, i, :, :])
            # print(hidden_input.shape)
            h_t, c_t = self.cell(hidden_input, h_t, c_t)

        
        return h_t, c_t
    

class CAE_LSTM_decoder(nn.Module):
    def __init__(self, hidden_size, latent_space):
        super(CAE_LSTM_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_space = latent_space

        self.cell = LSTM_cell(self.hidden_size, self.latent_space)

        # 使用 torch.Tensor 初始化参数张量
        self.init_c = Parameter(torch.Tensor(2*self.hidden_size, self.latent_space))
        self.init_h = Parameter(torch.Tensor(2*self.hidden_size, self.latent_space))
        self.b_h = Parameter(torch.Tensor(self.latent_space))
        self.b_c = Parameter(torch.Tensor(self.latent_space))

        self.decoder = Decoder(self.latent_space, 1)

        self.act = nn.ELU()

    def _init_states(self, h, c):
        combined_states = torch.cat((h, c), dim=1)
        combined_c = self.act(combined_states @ self.init_c+self.b_c)
        combined_h = self.act(combined_states @ self.init_h+self.b_h)

        return combined_h, combined_c
    
    def forward(self, h, c, seq_len): # input: (batch_size, hidden_size)
        h_t, c_t = self._init_states(h, c)

        output = []
        for _ in range(seq_len):
            h_t, c_t = self.cell(h, h_t, c_t)
            out = self.decoder(h_t)

            output.append(out)

        return torch.stack(output, dim=1) # output: (batch_size, seq_len, channel, output_size)
        
class CAE_LSTM_seq2seq(nn.Module):
    def __init__(self, latent_space, hidden_size):
        super(CAE_LSTM_seq2seq, self).__init__()
        self.latent_space = latent_space
        self.hidden_size = hidden_size

        self.encoder = CAE_LSTM_encoder(self.latent_space, self.hidden_size)
        self.decoder = CAE_LSTM_decoder(self.hidden_size, self.latent_space)

    def forward(self, x, seq_len, init_states=None):
        h, c = self.encoder(x, init_states)
        output = self.decoder(h, c, seq_len)

        return output