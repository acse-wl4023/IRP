import torch 
import torch.nn as nn

from kmeans_pytorch import kmeans

from torch_geometric.utils.repeat import repeat
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_mean
from torch_geometric.utils import remove_self_loops, coalesce
from torch_geometric.nn.conv import MessagePassing

from typing import List
from copy import deepcopy


# Edge pooling
def pool_edge_mean(cluster, edge_index, edge_attr):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1) 
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='mean')
    return edge_index, edge_attr

def avg_pool_mod_no_x(cluster, edge_index, edge_attr, batch, pos):
    cluster, perm = consecutive_cluster(cluster)

    # Pool edge attributes 
    edge_index_pool, edge_attr_pool = pool_edge_mean(cluster, edge_index, edge_attr)

    # Pool batch 
    batch_pool = None if batch is None else batch[perm]

    # Pool node positions 
    pos_pool = None if pos is None else scatter_mean(pos, cluster, dim=0)

    return edge_index_pool, edge_attr_pool, batch_pool, pos_pool, cluster, perm


class EdgeAggregator(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j, edge_attr):
        x_j = edge_attr
        return x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


class Encoder(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 n_mlp_mp,
                 num_mp_layers: List[int], 
                 lengthscales : List[float], 
                 bounding_box : List[float]):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.edge_aggregator = EdgeAggregator()
        self.hidden_channels = hidden_channels
        self.num_mp_layers = num_mp_layers
        self.depth = len(num_mp_layers)

        self.n_mlp_mp = n_mlp_mp # number of MLP layers in node/edge update functions used in message passing blocks
        self.lengthscales = lengthscales # lengthscales needed for voxel grid clustering
        self.l_char = [1.0] + self.lengthscales

        if not bounding_box:
            self.x_lo = None
            self.x_hi = None
            self.y_lo = None
            self.y_hi = None
        else:
            self.x_lo = bounding_box[0]
            self.x_hi = bounding_box[1]
            self.y_lo = bounding_box[2]
            self.y_hi = bounding_box[3]

        # assert(len(self.lengthscales) == self.depth), "size of lengthscales must be equal to size of n_mp_up"
        # ~~~~ DOWNWARD Message Passing
        # Edge updates: 
        self.edge_down_mps = torch.nn.ModuleList() 
        self.edge_down_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(self.depth):
            n_mp = self.num_mp_layers[m]
            edge_mp = torch.nn.ModuleList()
            edge_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*3
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                edge_mp.append(temp)
                edge_mp_norm.append( nn.LayerNorm(output_features) )

            self.edge_down_mps.append(edge_mp)
            self.edge_down_norms.append(edge_mp_norm)

        
        # Loop through levels: 
        self.node_down_mps = torch.nn.ModuleList() 
        self.node_down_norms = torch.nn.ModuleList()
        for m in range(self.depth):
            n_mp = self.num_mp_layers[m]
            node_mp = torch.nn.ModuleList()
            node_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*2
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                node_mp.append(temp)
                node_mp_norm.append( nn.LayerNorm(output_features) )

            self.node_down_mps.append(node_mp)
            self.node_down_norms.append(node_mp_norm)

        # For learned interpolations:
        self.edge_encoder_f2c_mlp = torch.nn.ModuleList()
        self.downsample_mlp = torch.nn.ModuleList()
        # self.downsample_norm = []

        # encoder for fine-to-coarse edge features 
        for j in range(self.n_mlp_mp):
            if j == 0: 
                input_features = 2 # 2-dimensional distance vector 
                output_features = hidden_channels
            else:
                input_features = hidden_channels
                output_features = hidden_channels
            self.edge_encoder_f2c_mlp.append( nn.Linear(input_features, output_features) )

        # downsample mlp  
        for j in range(self.n_mlp_mp):
            if j == 0:
                input_features = hidden_channels*2 # 2*hidden_channels for encoded f2c edges and sender node attributes 
                output_features = hidden_channels 
            else:
                input_features = hidden_channels
                output_features = hidden_channels
            self.downsample_mlp.append( nn.Linear(input_features, output_features) ) 
        self.downsample_norm = nn.LayerNorm(output_features)

        self.input_node_embedding = nn.Linear(self.input_channels, self.hidden_channels)
        self.input_edge_embedding = nn.Linear(self.input_channels, self.hidden_channels)
        self.act = nn.ELU()

    def forward(self, x, edge_index, edge_attr, pos, batch=None):

        # ~~~~ INITIAL MESSAGE PASSING ON FINE GRAPH (m = 0)
        m = 0 # level index 
        n_mp = self.num_mp_layers[m] # number of message passing blocks 
        x = self.input_node_embedding(x)
        x = self.act(x)

        edge_attr = self.input_edge_embedding(edge_attr)
        edge_attr = self.act(edge_attr)
        
        for i in range(n_mp):
            x_own = x[edge_index[0,:], :]
            x_nei = x[edge_index[1,:], :]
            edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)

            # edge update mlp
            for j in range(self.n_mlp_mp):
                edge_attr_t = self.edge_down_mps[m][i][j](edge_attr_t) 
                if j < self.n_mlp_mp - 1:
                    edge_attr_t = self.act(edge_attr_t)

            edge_attr = edge_attr + edge_attr_t
            edge_attr = self.edge_down_norms[m][i](edge_attr)

            # edge aggregation 
            edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

            x_t = torch.cat((x, edge_agg), axis=1)

            # node update mlp
            for j in range(self.n_mlp_mp):
                x_t = self.node_down_mps[m][i][j](x_t)
                if j < self.n_mlp_mp - 1:
                    x_t = self.act(x_t) 
            
            x = x + x_t
            x = self.node_down_norms[m][i](x)


        xs = [x] 
        edge_indices = [edge_index]
        # edge_attrs = [edge_attr]
        positions = [pos]
        batches = [batch]
        clusters = []
        edge_indices_f2c = []

        # ~~~~ Downward message passing 
        for m in range(1, self.depth):
            # Run voxel clustering
            cluster = tgnn.voxel_grid(pos = pos,
                                    size = self.lengthscales[m-1],
                                    batch = batch,
                                    start = [self.x_lo, self.y_lo], 
                                    end = [self.x_hi, self.y_hi])
            
            pos_f = pos.clone()
            edge_index, edge_attr, batch, pos, cluster, perm = avg_pool_mod_no_x(
                                                                            cluster,
                                                                            edge_index, 
                                                                            edge_attr,
                                                                            batch, 
                                                                            pos)
            
            n_nodes = x.shape[0]
            edge_index_f2c = torch.concat((torch.arange(0, n_nodes, dtype=torch.long, device=x.device).view(1,-1), cluster.view(1,-1)), axis=0)
            edge_indices_f2c += [edge_index_f2c]

            pos_c = pos
            edge_attr_f2c = (pos_c[edge_index_f2c[1,:]] - pos_f[edge_index_f2c[0,:]])/self.l_char[m-1]

            # encode the edge attributes with MLP
            for j in range(self.n_mlp_mp):
                edge_attr_f2c = self.edge_encoder_f2c_mlp[j](edge_attr_f2c)
                if j < self.n_mlp_mp - 1:
                    edge_attr_f2c = self.act(edge_attr_f2c)
                
            # append list
            # edge_attrs_f2c += [edge_attr_f2c]

            # Concatenate
            temp_ea = torch.cat((edge_attr_f2c, x), axis=1)

            # Apply downsample MLP
            for j in range(self.n_mlp_mp):
                temp_ea = self.downsample_mlp[j](temp_ea)
                if j < self.n_mlp_mp - 1:
                    temp_ea = self.act(temp_ea)
            
            temp_ea = edge_attr_f2c + temp_ea
            temp_ea = self.downsample_norm(temp_ea)

            x = self.edge_aggregator( (pos_f, pos_c), edge_index_f2c, temp_ea )

            # Append lists
            positions += [pos]
            batches += [batch]
            clusters += [cluster]

            # Do message passing on coarse graph
            for i in range(self.num_mp_layers[m]):
                x_own = x[edge_index[0,:], :]
                x_nei = x[edge_index[1,:], :]
                edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)

                for j in range(self.n_mlp_mp):
                    edge_attr_t = self.edge_down_mps[m][i][j](edge_attr_t) 
                    if j < self.n_mlp_mp - 1:
                        edge_attr_t = self.act(edge_attr_t)
                
                edge_attr = edge_attr + edge_attr_t
                edge_attr = self.edge_down_norms[m][i](edge_attr)
                edge_agg = self.edge_aggregator(x, edge_index, edge_attr)
                x_t = torch.cat((x, edge_agg), axis=1)

                for j in range(self.n_mlp_mp):
                    x_t = self.node_down_mps[m][i][j](x_t)
                    if j < self.n_mlp_mp - 1:
                        x_t = self.act(x_t) 
                
                x = x + x_t
                x = self.node_down_norms[m][i](x)

            xs += [x]
            edge_indices += [edge_index]
            # edge_attrs += [edge_attr]

        return x, edge_index, edge_attr, edge_indices, edge_indices_f2c, clusters, batches, positions
    


class Decoder(nn.Module):
    def __init__(self,
                 hidden_channels,
                 output_channels,
                 n_mlp_mp,
                 num_mp_layers: List[int]):
        super(Decoder, self).__init__()
        self.output_channels = output_channels
        self.edge_aggregator = EdgeAggregator()
        self.hidden_channels = hidden_channels
        self.num_mp_layers = num_mp_layers
        self.depth = len(num_mp_layers)


        self.n_mlp_mp = n_mlp_mp # number of MLP layers in node/edge update functions used in message passing blocks

        
        # ~~~~ UPWARD Message Passing
        self.edge_up_mps = torch.nn.ModuleList() 
        self.edge_up_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(self.depth):
            n_mp = self.num_mp_layers[m]
            edge_mp = torch.nn.ModuleList()
            edge_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*3
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                edge_mp.append(temp)
                edge_mp_norm.append( nn.LayerNorm(output_features) )

            self.edge_up_mps.append(edge_mp)
            self.edge_up_norms.append(edge_mp_norm)

        # Node updates: 
        self.node_up_mps = torch.nn.ModuleList()
        self.node_up_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(self.depth):
            n_mp = self.num_mp_layers[m]
            node_mp = torch.nn.ModuleList()
            node_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*2
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                node_mp.append(temp)
                node_mp_norm.append( nn.LayerNorm(output_features) )

            self.node_up_mps.append(node_mp)
            self.node_up_norms.append(node_mp_norm)

        self.edge_decoder_c2f_mlp = torch.nn.ModuleList()
        self.upsample_mlp = torch.nn.ModuleList()
        # self.upsample_norm = []

        # encoder for fine-to-coarse edge features 
        for j in range(self.n_mlp_mp):
            if j == 0: 
                input_features = 2 # 2-dimensional distance vector 
                output_features = hidden_channels
            else:
                input_features = hidden_channels
                output_features = hidden_channels
            self.edge_decoder_c2f_mlp.append( nn.Linear(input_features, output_features) )

        # upsample mlp
        for j in range(self.n_mlp_mp):
            if j == 0:
                input_features = hidden_channels*2 # 3 for encoded edge + sender and receiver node
                output_features = hidden_channels
            else:
                input_features = hidden_channels
                output_features = hidden_channels
            self.upsample_mlp.append( nn.Linear(input_features, output_features) )
        self.upsample_norm = nn.LayerNorm(output_features)

        self.output_node_embedding = nn.Linear(self.hidden_channels, self.output_channels)
        # self.output_edge_embedding = nn.Linear(self.hidden_channels, self.output_channels)
        self.act = nn.ELU()
    
    def forward(self, x, edge_index, edge_attr, edge_indices, edge_indices_f2c, clusters, batches, positions, lengthscales):
        l_char = [1.0] + lengthscales

        m = 0
        n_mp = self.num_mp_layers[m] # number of message passing blocks 
        edge_index = edge_indices[self.depth-1-m]
        
        for i in range(n_mp):
            x_own = x[edge_index[0,:], :]
            x_nei = x[edge_index[1,:], :]
            edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)

            # edge update mlp
            for j in range(self.n_mlp_mp):
                edge_attr_t = self.edge_up_mps[m][i][j](edge_attr_t) 
                if j < self.n_mlp_mp - 1:
                    edge_attr_t = self.act(edge_attr_t)

            edge_attr = edge_attr + edge_attr_t
            edge_attr = self.edge_up_norms[m][i](edge_attr)

            # edge aggregation 
            edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

            x_t = torch.cat((x, edge_agg), axis=1)

            # node update mlp
            for j in range(self.n_mlp_mp):
                x_t = self.node_up_mps[m][i][j](x_t)
                if j < self.n_mlp_mp - 1:
                    x_t = self.act(x_t) 
            
            x = x + x_t
            x = self.node_up_norms[m][i](x)

        for i in range(1, self.depth):
            # print('fuck')
            # interpolate
            pos_c = positions[self.depth-i].clone()
            pos_f = positions[self.depth-i-1].clone()
            edge_index_c2f = edge_indices_f2c[self.depth-i-1][[1, 0]].clone()
            # print(edge_indices_f2c[self.depth-i-1].shape)
            edge_attr_c2f = (pos_c[edge_index_c2f[0,:]] - pos_f[edge_index_c2f[1,:]])/l_char[self.depth-i-1]

            for j in range(self.n_mlp_mp):
                edge_attr_c2f = self.edge_decoder_c2f_mlp[j](edge_attr_c2f)
                edge_attr_c2f = self.act(edge_attr_c2f)

            # print(self.depth)
            x = x[clusters[self.depth-i-1]]
            
            # 得到c2f的边的特征
            temp_ea = torch.cat((edge_attr_c2f, x), axis=1)

            for j in range(self.n_mlp_mp):
                temp_ea = self.upsample_mlp[j](temp_ea)
                temp_ea = self.act(temp_ea)
            temp_ea = edge_attr_c2f + temp_ea
            temp_ea = self.upsample_norm(temp_ea)

            # 通过消息传递得到精细图的节点特征
            x = self.edge_aggregator((pos_c, pos_f), edge_index_c2f, temp_ea)


            # message passing
            n_mp = self.num_mp_layers[m] # number of message passing blocks
            edge_index = edge_indices[self.depth-1-m]
            for i in range(n_mp):
                x_own = x[edge_index[0,:], :]
                x_nei = x[edge_index[1,:], :]
                edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)

                # edge update mlp
                for j in range(self.n_mlp_mp):
                    edge_attr_t = self.edge_up_mps[m][i][j](edge_attr_t) 
                    if j < self.n_mlp_mp - 1:
                        edge_attr_t = self.act(edge_attr_t)

                edge_attr = edge_attr + edge_attr_t
                edge_attr = self.edge_up_norms[m][i](edge_attr)

                # edge aggregation 
                edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

                x_t = torch.cat((x, edge_agg), axis=1)

                # node update mlp
                for j in range(self.n_mlp_mp):
                    x_t = self.node_up_mps[m][i][j](x_t)
                    if j < self.n_mlp_mp - 1:
                        x_t = self.act(x_t) 
                
                x = x + x_t
                x = self.node_up_norms[m][i](x)

        x = self.output_node_embedding(x)
        x = self.act(x)
        # edge_attr = self.output_edge_embedding(edge_attr)
        # edge_attr = self.act(edge_attr)

        return x, edge_index
    

class GAE(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 output_channels,
                 n_mlp_mp,
                 num_mp_layers: List[int], 
                 lengthscales : List[float], 
                 bounding_box : List[float]):
        super(GAE, self).__init__()

        
        self.lengthscales = lengthscales
        self.encoder = Encoder(input_channels, hidden_channels, n_mlp_mp, num_mp_layers, lengthscales, bounding_box)
        self.decoder = Decoder(hidden_channels, output_channels, n_mlp_mp, num_mp_layers)

    def forward(self, x, edge_index, edge_attr, pos, batch=None):
        x, edge_index, edge_attr, edge_indices, edge_indices_f2c, clusters, batches, positions = self.encoder(x, edge_index, edge_attr, pos, batch)
        x, edge_index = self.decoder(x, edge_index, edge_attr, edge_indices, edge_indices_f2c, clusters, batches, positions, self.lengthscales)

        return x, edge_index



        

            


