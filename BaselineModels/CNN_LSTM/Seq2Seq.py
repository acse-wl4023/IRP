import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GConvLSTMCell(torch.nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super(GConvLSTMCell, self).__init__()
        self.input_dim = input_channel
        self.hidden_dim = hidden_channel
        self.conv = GCNConv(self.input_dim+self.hidden_dim, 4*self.hidden_dim)

    def forward(self, x, edge_index, h_cur, c_cur, edge_weight = None):
        combined = torch.cat((x, h_cur), axis=1)
        combined_conv = self.conv(combined, edge_index, edge_weight)
        cc_f,cc_i,cc_o,cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f*c_cur + i*g
        h_next = o*torch.tanh(c_next)

        return h_next, c_next
    

class Encoder(nn.Module):
    def __init__(self, seq_len, input_channels, hidden_channels, device):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.device = device

        cellList = []
        for _ in range(seq_len):
            cellList.append(GConvLSTMCell(self.input_channels, self.hidden_channels))
        
        self.cellList = nn.ModuleList(cellList)
    
    def init_hidden_state(self, num_nodes):
        h = torch.zeros((num_nodes, self.hidden_channels), dtype=torch.float, device=self.device)
        c = torch.zeros((num_nodes, self.hidden_channels), dtype=torch.float, device=self.device)
        return h, c

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes= x.shape[1]
        h_cur, c_cur = self.init_hidden_state(num_nodes)
        output_hList = []
        for i in range(self.seq_len):
            h_cur, c_cur = self.cellList[i](x[i, :, :], edge_index, h_cur, c_cur, edge_weight)
            output_hList.append(h_cur)

        output = torch.stack(output_hList, dim=0)
        return output, (h_cur, c_cur)


class Decoder(nn.Module):
    def __init__(self, seq_len, hidden_channels, output_channels):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

        cellList = []
        for _ in range(seq_len):
            cellList.append(GConvLSTMCell(self.hidden_channels, self.output_channels))
        
        self.cellList = nn.ModuleList(cellList)

        self.conv_h = GCNConv(self.hidden_channels, self.output_channels)
        self.conv_c = GCNConv(self.hidden_channels, self.output_channels)


    def forward(self, x, c, edge_index, edge_weight=None):
        h_cur = self.conv_h(x, edge_index, edge_weight)
        c_cur = self.conv_c(c, edge_index, edge_weight)

        output_hList = []
        for i in range(self.seq_len):
            h_cur, c_cur = self.cellList[i](x, edge_index, h_cur, c_cur, edge_weight)
            output_hList.append(h_cur)

        output = torch.stack(output_hList, dim=0)
        return output, (h_cur, c_cur)


class GConvLSTM(nn.Module):
    def __init__(self, input_seq_len, output_seq_len, channels, hidden_channels, device):
        super(GConvLSTM, self).__init__()
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.device = device

        self.encoder = Encoder(self.input_seq_len, self.channels, self.hidden_channels, self.device)
        self.decoder = Decoder(self.output_seq_len, self.hidden_channels, self.channels)

        self.leakyRelu = nn.LeakyReLU(0.01)
    def forward(self, x, edge_index, edge_weight):
        _, (x, c_cur) = self.encoder(x, edge_index, edge_weight)
        output, _ = self.decoder(x, c_cur, edge_index, edge_weight)
        output = self.leakyRelu(output)
        return output
    