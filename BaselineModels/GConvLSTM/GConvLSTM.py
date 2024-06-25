import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GConvLSTMCell(torch.nn.Module):
    def __init__(self, input_feature, hidden_dim):
        super(GConvLSTMCell, self).__init__()
        self.input_dim = input_feature
        self.hidden_dim = hidden_dim
        self.conv = GCNConv(self.input_dim, 4*self.hidden_dim)

    def forward(self, x, edge_index, h_cur, c_cur):
        combined = x+h_cur
        combined_conv = self.conv(combined, edge_index)
        cc_f,cc_i,cc_o,cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        return cc_f
