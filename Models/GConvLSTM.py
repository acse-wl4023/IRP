import torch
import torch_geometric.nn as tgnn
import torch.nn as nn

class GConvLSTM_cell(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super(GConvLSTM_cell, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        
        self.conv = tgnn.GCNConv(in_channels=self.input_channel + self.hidden_channel,
                                 out_channels=4*hidden_channel)

        # 初始化参数
        nn.init.xavier_normal_(self.conv.lin.weight)


    def forward(self, x, edge_index, edge_attr, h_t, c_t): # input = (input_size, channel)

        combined = torch.cat([x, h_t], dim=1)
        # print(combined.shape)
        combined_conv = self.conv(combined, edge_index, edge_attr)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channel, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_t + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class GConvLSTM_encoder(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super(GConvLSTM_encoder, self).__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel

        self.cell = GConvLSTM_cell(self.input_channel, self.hidden_channel)

    def _init_states(self, x):
        h_t = torch.zeros(x.size(1), self.hidden_channel, dtype=x.dtype).to(x.device)
        c_t = torch.zeros(x.size(1), self.hidden_channel, dtype=x.dtype).to(x.device)
        return h_t, c_t
    
    def forward(self, x, edge_index, edge_attr, init_states=None): # input = (seq_length, input_size, channels):
        # 状态初始化
        if init_states is None:
            h_t, c_t = self._init_states(x)
        else:
            h_t, c_t = init_states

        seq_len = x.size(0)

        for i in range(seq_len):
            # print(x[i, :, :].shape), h_t.shape
            h_t, c_t = self.cell(x[i, :, :], edge_index, edge_attr, h_t, c_t)

        
        return h_t, c_t
    

class GConvLSTM_decoder(nn.Module):
    def __init__(self, hidden_channel, output_channel):
        super(GConvLSTM_decoder, self).__init__()
        self.hidden_channel = hidden_channel
        self.output_channel = output_channel

        self.cell = GConvLSTM_cell(self.hidden_channel, self.output_channel)

        
        self.conv = tgnn.GCNConv(2*self.hidden_channel, 2*self.output_channel)

        self.act = nn.ELU()

    def _init_states(self, h, c, edge_index, edge_attr):
        combined_states = torch.cat((h, c), dim=1)

        states = self.act(self.conv(combined_states, edge_index, edge_attr))
        combined_h, combined_c = torch.split(states, self.output_channel, dim=1)


        return combined_h, combined_c
    
    def forward(self, h, c, edge_index, edge_attr, seq_len): # input: (batch_size, hidden_size)
        h_t, c_t = self._init_states(h, c, edge_index, edge_attr)
        output = []
        for _ in range(seq_len):
            h_t, c_t = self.cell(h, edge_index, edge_attr, h_t, c_t)
            
            output.append(h_t)

        return torch.stack(output, dim=0) # output: (batch_size, seq_len, channel, output_size)
        
    
class GConvLSTM_seq2seq(nn.Module):
    def __init__(self, input_channels, hidden_channel, output_channel):
        super(GConvLSTM_seq2seq, self).__init__()
        self.input_channels = input_channels
        self.hidden_channel = hidden_channel
        self.output_channel = output_channel

        self.encoder = GConvLSTM_encoder(self.input_channels, self.hidden_channel)
        self.decoder = GConvLSTM_decoder(self.hidden_channel, self.output_channel)

    def forward(self, x, edge_index, edge_attr, seq_len, init_states=None):
        h, c = self.encoder(x, edge_index, edge_attr, init_states)
      
        output = self.decoder(h, c, edge_index, edge_attr, seq_len)

        return output

    


