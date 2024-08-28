import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch.nn import Parameter
from .GAE_no_edge_attr import Encoder, Decoder
from .GConvLSTM import GConvLSTM_cell

class GAE_GConvLSTM_encoder(nn.Module):
    def __init__(self, latent_space,
                 input_node_channel,
                 num_mp_layers, 
                 clusters,
                 centroids,
                 hidden_channels: int,
                 n_mlp_mp: int,
                 hidden_channel_lstm):
        super(GAE_GConvLSTM_encoder, self).__init__()

        self.latent_space = latent_space # output of the cae encoder, input of the lstm encoder
        self.input_node_channel = input_node_channel
        self.num_mp_layers = num_mp_layers
        self.clusters = clusters
        self.centroids = centroids
        self.hidden_channels = hidden_channels
        self.n_mlp_mp = n_mlp_mp
        self.hidden_channel_lstm = hidden_channel_lstm

        self.cell = GConvLSTM_cell(self.hidden_channels, self.hidden_channel_lstm)

        self.encoder = Encoder(self.input_node_channel,
                               self.num_mp_layers,
                               self.clusters,
                               self.centroids,
                               self.hidden_channels,
                               self.n_mlp_mp)

    def _init_states(self, x):
        h_t = torch.zeros(self.latent_space, self.hidden_channel_lstm, dtype=x.dtype).to(x.device)
        c_t = torch.zeros(self.latent_space, self.hidden_channel_lstm, dtype=x.dtype).to(x.device)
        return h_t, c_t
    
    def forward(self, x, edge_index, edge_attr, pos, init_states=None): # input = (seq_length, channels, input_size):
        # 状态初始化
        if init_states is None:
            h_t, c_t = self._init_states(x)
        else:
            h_t, c_t = init_states

        seq_len = x.size(0)
        # print(seq_len)

        for i in range(seq_len):
            hidden_input, hidden_edge_index, hidden_edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters = self.encoder(x[i, :, :], edge_index, edge_attr, pos)
            
            h_t, c_t = self.cell(hidden_input, hidden_edge_index, hidden_edge_attr, h_t, c_t)

        
        return h_t, c_t, hidden_edge_index, hidden_edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters
    

class GAE_GConvLSTM_decoder(nn.Module):
    def __init__(self,
                 hidden_channel_lstm, latent_space,
                 output_node_channel,
                 num_mp_layers,
                 hidden_channels,
                 n_mlp_mp):
        super(GAE_GConvLSTM_decoder, self).__init__()
        self.hidden_channel_lstm = hidden_channel_lstm
        self.latent_space = latent_space

        self.output_node_channel = output_node_channel
        self.num_mp_layers = num_mp_layers
        self.hidden_channels = hidden_channels
        self.n_mlp_mp = n_mlp_mp

        self.cell = GConvLSTM_cell(self.hidden_channel_lstm, self.hidden_channels)

        self.init = tgnn.GCNConv(2*self.hidden_channel_lstm, 2*self.hidden_channels)

        self.decoder = Decoder(self.output_node_channel,
                               self.num_mp_layers,
                               self.hidden_channels,
                               self.n_mlp_mp)

        self.act = nn.ELU()

    def _init_states(self, h, c, edge_index, edge_attr):
        combined_states = torch.cat((h, c), dim=1)

        states = self.act(self.init(combined_states, edge_index, edge_attr))
        combined_h, combined_c = torch.split(states, self.hidden_channels, dim=1)

        return combined_h, combined_c
    
    def forward(self, h, c, seq_len, 
                hidden_edge_index, 
                hidden_edge_attr, 
                edge_indices, 
                edge_attrs, 
                edge_indices_f2c, 
                position, 
                node_attrs, 
                clusters): # input: (hidden_channels, hidden_size)
        # if torch.isnan(h).any():
        #     print(33333)
        h_t, c_t = self._init_states(h, c, hidden_edge_index, hidden_edge_attr)


        output = []
        for _ in range(seq_len):
            
            h_t, c_t = self.cell(h, hidden_edge_index, hidden_edge_attr, h_t, c_t)

            out, _, _ = self.decoder(h_t,
                                    hidden_edge_index,
                                    hidden_edge_attr,
                                    edge_indices,
                                    edge_attrs,
                                    edge_indices_f2c,
                                    position,
                                    node_attrs,
                                    clusters)

            output.append(out)

        return torch.stack(output, dim=0) # output: (seq_len, output_size, channel)
    

class GAE_GConvLSTM_seq2seq(nn.Module):
    def __init__(self, latent_space,
                       hidden_channel_lstm,
                       input_node_channel: int,
                       output_node_channel: int,
                       num_mp_layers,
                       clusters,
                       centroids,
                       hidden_channels: int,
                       n_mlp_mp: int):

        super(GAE_GConvLSTM_seq2seq, self).__init__()
        self.latent_space = latent_space
        self.hidden_channel_lstm = hidden_channel_lstm

        self.input_node_channel = input_node_channel
        self.output_node_channel = output_node_channel
        self.num_mp_layers = num_mp_layers
        self.clusters = clusters
        self.centroids = centroids
        self.hidden_channels = hidden_channels
        self.n_mlp_mp = n_mlp_mp

        self.encoder = GAE_GConvLSTM_encoder(self.latent_space,
                                             self.input_node_channel,
                                             self.num_mp_layers,
                                             self.clusters,
                                             self.centroids,
                                             self.hidden_channels,
                                             self.n_mlp_mp,
                                             self.hidden_channel_lstm)
        
        self.decoder = GAE_GConvLSTM_decoder(self.hidden_channel_lstm,
                                             self.latent_space,
                                             self.output_node_channel,
                                             self.num_mp_layers,
                                             self.hidden_channels,
                                             self.n_mlp_mp)

    def forward(self, x, seq_len, edge_index, edge_attr, pos, init_states=None):
        h_t, c_t, hidden_edge_index, hidden_edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters = self.encoder(x, edge_index, edge_attr, pos, init_states)
        # print(h_t.shape, c_t.shape)

        output = self.decoder(h_t, c_t, seq_len, hidden_edge_index, hidden_edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters)

        return output