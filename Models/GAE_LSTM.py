import torch
import torch.nn as nn
from torch.nn import Parameter
from .GAE_no_edge_attr import Encoder, Decoder
from .LSTM_Cell import LSTM_cell



class GAE_LSTM_encoder(nn.Module):
    def __init__(self, latent_space,
                 input_node_channel,
                 num_mp_layers, 
                 clusters,
                 centroids,
                 hidden_channels: int,
                 n_mlp_mp: int,
                 hidden_size):
        super(GAE_LSTM_encoder, self).__init__()

        self.latent_space = latent_space # output of the cae encoder, input of the lstm encoder
        self.input_node_channel = input_node_channel
        self.num_mp_layers = num_mp_layers
        self.clusters = clusters
        self.centroids = centroids
        self.hidden_channels = hidden_channels
        self.n_mlp_mp = n_mlp_mp
        self.hidden_size = hidden_size # input of the lstm encoder

        self.cell = LSTM_cell(self.latent_space, self.hidden_size)

        self.encoder = Encoder(self.input_node_channel,
                               self.num_mp_layers,
                               self.clusters,
                               self.centroids,
                               self.hidden_channels,
                               self.n_mlp_mp)

    def _init_states(self, x):
        h_t = torch.zeros(self.hidden_channels, self.hidden_size, dtype=x.dtype).to(x.device)
        c_t = torch.zeros(self.hidden_channels, self.hidden_size, dtype=x.dtype).to(x.device)
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
            h_t, c_t = self.cell(hidden_input.permute(1, 0), h_t, c_t)

        
        return h_t, c_t, hidden_edge_index, hidden_edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters
    

class GAE_LSTM_decoder(nn.Module):
    def __init__(self,
                 hidden_size, latent_space,
                 output_node_channel,
                 num_mp_layers,
                 hidden_channels,
                 n_mlp_mp):
        super(GAE_LSTM_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_space = latent_space

        self.output_node_channel = output_node_channel
        self.num_mp_layers = num_mp_layers
        self.hidden_channels = hidden_channels
        self.n_mlp_mp = n_mlp_mp

        self.cell = LSTM_cell(self.hidden_size, self.latent_space)

        # 使用 torch.Tensor 初始化参数张量
        self.init_c = Parameter(torch.Tensor(2*self.hidden_size, self.latent_space))
        self.init_h = Parameter(torch.Tensor(2*self.hidden_size, self.latent_space))
        self.b_h = Parameter(torch.Tensor(self.latent_space))
        self.b_c = Parameter(torch.Tensor(self.latent_space))

        # 使用 torch.Tensor 初始化参数张量
        # self.init_c = nn.Sequential(nn.Linear(2*self.hidden_size, self.latent_space),
        #                             nn.Dropout1d())
        # self.init_h = nn.Sequential(nn.Linear(2*self.hidden_size, self.latent_space),
        #                             nn.Dropout1d())

        # self.init_c = nn.Linear(2*self.hidden_size, self.latent_space)
        # self.init_h = nn.Linear(2*self.hidden_size, self.latent_space)

        self.decoder = Decoder(self.output_node_channel,
                               self.num_mp_layers,
                               self.hidden_channels,
                               self.n_mlp_mp)

        self.act = nn.ELU()
        # 初始化参数
        self._initialize_weights() # 只针对latent space = 25

    def _initialize_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def _init_states(self, h, c):
        combined_states = torch.cat((h, c), dim=1)
        # print(combined_states.shape)
        combined_c = self.act(combined_states @ self.init_c+self.b_c)
        combined_h = self.act(combined_states @ self.init_h+self.b_h)

        # combined_c = self.act(self.init_c(combined_states))
        # combined_h = self.act(self.init_h(combined_states))

        # combined_c = torch.zeros(h.size(0), self.latent_space, dtype=h.dtype).to(h.device)
        # combined_h = torch.zeros(h.size(0), self.latent_space, dtype=h.dtype).to(h.device)

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
        h_t, c_t = self._init_states(h, c)
        # if torch.isnan(h_t).any():
        #     print(11111)

        output = []
        for _ in range(seq_len):
            if torch.isnan(h_t).any():
                print(22222)
            h_t, c_t = self.cell(h, h_t, c_t)

            out, _, _ = self.decoder(h_t.permute(1, 0),
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
        
class GAE_LSTM_seq2seq(nn.Module):
    def __init__(self, latent_space, hidden_size,
                 input_node_channel: int,
                 output_node_channel: int,
                 num_mp_layers,
                 clusters,
                 centroids,
                 hidden_channels: int,
                 n_mlp_mp: int):
        super(GAE_LSTM_seq2seq, self).__init__()
        self.latent_space = latent_space
        self.hidden_size = hidden_size

        self.input_node_channel = input_node_channel
        self.output_node_channel = output_node_channel
        self.num_mp_layers = num_mp_layers
        self.clusters = clusters
        self.centroids = centroids
        self.hidden_channels = hidden_channels
        self.n_mlp_mp = n_mlp_mp

        self.encoder = GAE_LSTM_encoder(self.latent_space,
                                        self.input_node_channel,
                                        self.num_mp_layers,
                                        self.clusters,
                                        self.centroids,
                                        self.hidden_channels,
                                        self.n_mlp_mp,
                                        self.hidden_size)
        
        self.decoder = GAE_LSTM_decoder(self.hidden_size,
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