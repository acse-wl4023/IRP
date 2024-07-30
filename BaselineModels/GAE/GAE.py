import torch 
import torch.nn as nn

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


def avg_pool_mod_no_x(cluster, edge_index, edge_attr, pos):
    cluster, perm = consecutive_cluster(cluster)
    # Pool edge attributes 
    edge_index_pool, edge_attr_pool = pool_edge_mean(cluster, edge_index, edge_attr)
    # Pool node positions 
    pos_pool = None if pos is None else scatter_mean(pos, cluster, dim=0)
    return edge_index_pool, edge_attr_pool, pos_pool, cluster, perm


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
                 input_node_channel: int,
                 bounding_box: List[float],
                 lengthscales: List[float],
                 num_mp_layers: List[int],
                 hidden_channels: int,
                 n_mlp_mp: int):
        super(Encoder, self).__init__()

        self.input_node_channel = input_node_channel
        self.hidden_channels = hidden_channels
        self.edge_aggregator = EdgeAggregator()

        # bounding box for voxel clustering
        self.x_lo = bounding_box[0]
        self.x_hi = bounding_box[1]
        self.y_lo = bounding_box[2]
        self.y_hi = bounding_box[3]

        # define the size of voxel
        self.lengthscales = lengthscales
        self.l_char = [1.0] + self.lengthscales

        # number of message passing layers in each blocks
        self.num_mp_layers = num_mp_layers

        # depth of the encoder
        self.depth = len(self.num_mp_layers)

        # activate function
        self.act = nn.ELU()

        # List that store the message passing blocks.
        self.mp_blocks = nn.ModuleList()
        for i in range(self.depth):
            block = nn.ModuleList() # one block contains multiple mp layers
            for j in range(self.num_mp_layers[i]):
                block.append(GCNConv(self.hidden_channels, self.hidden_channels))
            
            self.mp_blocks.append(block)

        # the MLP for the edge that conect the fine graph and the coarsen graph
         # encoder for fine-to-coarse edge features 
        self.n_mlp_mp = n_mlp_mp

        # 用于生成f2c的边特征
        self.edge_encoder_f2c_mlp = nn.ModuleList()
        for j in range(self.n_mlp_mp):
            if j == 0: 
                input_features = 2 # 2-dimensional distance vector 
                output_features = self.hidden_channels
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.edge_encoder_f2c_mlp.append(nn.Linear(input_features, output_features))

        # 用于节点的特征嵌入
        self.node_embeding_mlp = nn.ModuleList()
        for j in range(self.n_mlp_mp):
            if j == 0: 
                input_features = self.input_node_channel # 2-dimensional distance vector 
                output_features = self.hidden_channels
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.node_embeding_mlp.append(nn.Linear(input_features, output_features))
        
        # 用于f2c的过程中将sender根据edge_f2c的特征进行编码，之后进行edge_aggregate
        self.downsample_mlp = nn.ModuleList()
        for j in range(self.n_mlp_mp):
            if j == 0: 
                input_features = self.hidden_channels*2 # 2-dimensional distance vector 
                output_features = self.hidden_channels
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.downsample_mlp.append(nn.Linear(input_features, output_features))
            self.downsample_norm = nn.LayerNorm(output_features)
        
        
        self.edge_encoder_mlp = nn.ModuleList()
        for i in range(self.depth):
            mlp = nn.ModuleList() # one block contains multiple mp layers
            for j in range(self.n_mlp_mp):
                if j == 0: 
                    input_features = 1 # 2-dimensional distance vector 
                    output_features = self.hidden_channels
                elif j == self.n_mlp_mp-1:
                    input_features = self.hidden_channels
                    output_features = 1
                else:
                    input_features = self.hidden_channels
                    output_features = self.hidden_channels
                # print(input_features, output_features)
                mlp.append(nn.Linear(input_features, output_features))
            self.edge_encoder_mlp.append(mlp)


    def forward(self, x, edge_index, edge_attr, pos):
        '''
        1. message passing(maybe GCN)
        2. voxel cluster
        '''
        # feature embedding
        for j in range(self.n_mlp_mp):
            x = self.node_embeding_mlp[j](x)
            x = self.act(x)

        # initialize 进行第一次图卷积
        # initialize 对最粗的图进行卷积
        
        edge_attr = edge_attr.fill_(1).unsqueeze(1)
        # use mlp update the attr of edge of coarse graph
        for j in range(self.n_mlp_mp):
            edge_attr = self.edge_encoder_mlp[0][j](edge_attr)
            edge_attr = self.act(edge_attr)
        
        for j in range(self.num_mp_layers[0]):
            x = self.mp_blocks[0][j](x, edge_index, edge_attr)
            x = self.act(x)
        
        # 储存f2c的边
        edge_indices_f2c = []
        edge_indices = [edge_index]
        node_attrs = [x]
        edge_attrs = [edge_attr]
        position = [pos] # 存储生成的每张图的节点的坐标 从最精细到最粗糙
        clusters = []

        for i in range(1, self.depth):
            '''
            图粗化
            '''
            # Run voxel clustering
            pos_cpu = pos.cpu()
            cluster = tgnn.pool.voxel_grid(pos = pos_cpu,
                                           size = self.lengthscales[i-1],
                                           start = [self.x_lo, self.y_lo],
                                           end = [self.x_hi, self.y_hi])
            cluster = cluster.to(pos.device)
            pos_fine = deepcopy(pos)

            # 计算得到新的图的边
            edge_index, edge_attr, pos, cluster, perm = avg_pool_mod_no_x(cluster,
                                                                          edge_index,
                                                                          edge_attr,
                                                                          pos)
            
            edge_attr.fill_(1)
            # use mlp update the attr of edge of coarse graph
    
            for j in range(self.n_mlp_mp):
                # print('1111')
                edge_attr = self.edge_encoder_mlp[i][j](edge_attr)
                edge_attr = self.act(edge_attr)

            position.append(pos)
            clusters.append(cluster)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)

            n_nodes = x.shape[0]
            edge_index_f2c = torch.concat( (torch.arange(0, n_nodes, dtype=torch.long, device=x.device).view(1,-1), cluster.view(1,-1)), axis=0 )
            edge_indices_f2c.append(edge_index_f2c)
            pos_coarse = pos

            # 在fine graph和coarsen graph间构建连接，用于计算coarsen graph节点的特征值
            # coarsen graph的节点是voxel的中点，利用距离来计算权重
            edge_attr_f2c = (pos_coarse[edge_index_f2c[1,:]] - pos_fine[edge_index_f2c[0,:]])/self.l_char[i-1]
            # print(pos_coarse.dtype)
            # print(edge_attr_f2c.shape)
            # encode the edge attributes with MLP
            for j in range(self.n_mlp_mp):
                # print(edge_attr_f2c.dtype)
                edge_attr_f2c = self.edge_encoder_f2c_mlp[j](edge_attr_f2c)
                edge_attr_f2c = self.act(edge_attr_f2c)
        
            # message passing to get the coarsen graph node attribution
            temp_ea = torch.cat((edge_attr_f2c, x), axis=1)
            for j in range(self.n_mlp_mp):
                temp_ea = self.downsample_mlp[j](temp_ea)
                temp_ea = self.act(temp_ea)
            temp_ea = edge_attr_f2c + temp_ea
            temp_ea = self.downsample_norm(temp_ea)
            x = self.edge_aggregator((pos_fine, pos_coarse), edge_index_f2c, temp_ea)

            for j in range(self.num_mp_layers[i]):
                x = self.mp_blocks[i][j](x, edge_index, edge_attr)
                x = self.act(x)
            node_attrs.append(x)
        return x, edge_index, edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters
    

class Decoder(nn.Module):
    def __init__(self, 
                 output_node_channel: int,
                 lengthscales: List[float],
                 num_mp_layers: List[int],
                 hidden_channels: int,
                 n_mlp_mp: int):
        super(Decoder, self).__init__()

        self.depth = len(num_mp_layers)
        self.output_node_channel = output_node_channel
        self.num_mp_layer = num_mp_layers
        self.hidden_channels = hidden_channels
        self.n_mlp_mp = n_mlp_mp
        self.num_mp_layers = num_mp_layers
        self.edge_aggregator = EdgeAggregator()

        # define the size of voxel
        self.lengthscales = lengthscales
        self.l_char = [1.0] + self.lengthscales
        
        # activate function
        self.act = nn.ELU()

        # List that store the message passing blocks.
        self.mp_blocks = nn.ModuleList()
        for i in range(self.depth):
            block = nn.ModuleList() # one block contains multiple mp layers
            for j in range(self.num_mp_layers[i]):
                block.append(GCNConv(self.hidden_channels, self.hidden_channels))
            
            self.mp_blocks.append(block)

        # 用于生成c2f的边特征
        self.edge_decoder_c2f_mlp = nn.ModuleList()
        for j in range(self.n_mlp_mp):
            if j == 0: 
                input_features = 2 # 2-dimensional distance vector 
                output_features = self.hidden_channels
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.edge_decoder_c2f_mlp.append(nn.Linear(input_features, output_features))

        # 用于c2f的过程中将sender根据edge_f2c的特征进行编码，之后进行edge_aggregate
        self.upsample_mlp = nn.ModuleList()
        for j in range(self.n_mlp_mp):
            if j == 0: 
                input_features = self.hidden_channels*2 # 2-dimensional distance vector 
                output_features = self.hidden_channels
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.upsample_mlp.append(nn.Linear(input_features, output_features))
            self.upsample_norm = nn.LayerNorm(output_features)
        
        # 用于生成fine graph的边特征
        # self.edge_decoder_mlp = nn.ModuleList()
        # for j in range(self.n_mlp_mp):
        #     if j == 0: 
        #         input_features = 1 # 2-dimensional distance vector 
        #         output_features = self.hidden_channels
        #     elif j == self.n_mlp_mp-1:
        #         input_features = self.hidden_channels
        #         output_features = 1
        #     else:
        #         input_features = self.hidden_channels
        #         output_features = self.hidden_channels
        #     self.edge_decoder_mlp.append(nn.Linear(input_features, output_features))
        self.edge_decoder_mlp = nn.ModuleList()
        for i in range(self.depth):
            mlp = nn.ModuleList() # one block contains multiple mp layers
            for j in range(self.n_mlp_mp):
                if j == 0: 
                    input_features = 1 # 2-dimensional distance vector 
                    output_features = self.hidden_channels
                elif j == self.n_mlp_mp-1:
                    input_features = self.hidden_channels
                    output_features = 1
                else:
                    input_features = self.hidden_channels
                    output_features = self.hidden_channels
                mlp.append(nn.Linear(input_features, output_features))
            self.edge_decoder_mlp.append(mlp)

        # 用于节点的特征嵌入
        self.node_decoder_mlp = nn.ModuleList()
        for j in range(self.n_mlp_mp):
            if j != self.n_mlp_mp-1: 
                input_features = self.hidden_channels # 2-dimensional distance vector 
                output_features = self.hidden_channels
            else:
                input_features = self.hidden_channels
                output_features = self.output_node_channel
            self.node_decoder_mlp.append(nn.Linear(input_features, output_features))

    def forward(self, x, edge_index, edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters):
        # initialize 对最粗的图进行卷积
        edge_attr.fill_(1)
        # use mlp update the attr of edge of coarse graph
        for j in range(self.n_mlp_mp):
            edge_attr = self.edge_decoder_mlp[0][j](edge_attr)
            edge_attr = self.act(edge_attr)

        for j in range(self.num_mp_layers[0]):
            x = self.mp_blocks[0][j](x, edge_index, edge_attr)
            x = self.act(x)

        for i in range(self.depth-1, 0, -1):
            '''
            粗图细化
            '''
            # 分别获取粗细图的节点坐标和连接他们的边的属性
            pos_fine = position[i-1]
            pos_coarse = position[i]
            edge_index_c2f = edge_indices_f2c[i-1][[1, 0]] # 上下交换

            edge_attr_c2f = (pos_coarse[edge_index_c2f[0,:]] - pos_fine[edge_index_c2f[1,:]])/self.l_char[i]
            # encode the edge attributes with MLP
            for j in range(self.n_mlp_mp):
                edge_attr_c2f = self.edge_decoder_c2f_mlp[j](edge_attr_c2f)
                edge_attr_c2f = self.act(edge_attr_c2f)

            # message passing to get the coarsen graph node attribution
            x = x[clusters[i-1]]

            temp_ea = torch.cat((edge_attr_c2f, x), axis=1)

            for j in range(self.n_mlp_mp):
                temp_ea = self.upsample_mlp[j](temp_ea)
                temp_ea = self.act(temp_ea)
            temp_ea = edge_attr_c2f + temp_ea
            temp_ea = self.upsample_norm(temp_ea)
            x = self.edge_aggregator((pos_coarse, pos_fine), edge_index_c2f, temp_ea)

            edge_index = edge_indices[i-1]
            edge_attr = edge_attrs[i-1]
            edge_attr.fill_(1)
        # use mlp update the attr of edge of coarse graph
            for j in range(self.n_mlp_mp):
                edge_attr = self.edge_decoder_mlp[i][j](edge_attr)
                edge_attr = self.act(edge_attr)
            
            for j in range(self.num_mp_layer[i]):
                x = self.mp_blocks[i][j](x, edge_index, edge_attr)
                x = self.act(x)
            
        for j in range(self.n_mlp_mp):
            x = self.node_decoder_mlp[j](x)
            x = self.act(x)
        return x, edge_index, edge_attr
    

class GAE(nn.Module):
    def __init__(self, 
                 input_node_channel: int,
                 output_node_channel: int,
                 bounding_box: List[float],
                 lengthscales: List[float],
                 num_mp_layers: List[int],
                 hidden_channels: int,
                 n_mlp_mp: int):
        super(GAE, self).__init__()
        self.input_node_channel = input_node_channel
        self.output_node_channel = output_node_channel
        self.bounding_box = bounding_box
        self.lengthscales = lengthscales
        self.num_mp_layers = num_mp_layers
        self.hidden_channels = hidden_channels
        self.n_mlp_mp = n_mlp_mp

        self.encoder = Encoder(self.input_node_channel,
                               self.bounding_box,
                               self.lengthscales,
                               self.num_mp_layers,
                               self.hidden_channels,
                               self.n_mlp_mp)
        self.decoder = Decoder(self.output_node_channel,
                               self.lengthscales,
                               self.num_mp_layers,
                               self.hidden_channels,
                               self.n_mlp_mp)
    
    def forward(self, x, edge_index, edge_attr, pos):
        x, edge_index, edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters = self.encoder(x, edge_index, edge_attr, pos)
        # print(x.max(), x.min())
        x, edge_index, edge_attr = self.decoder(x, edge_index, edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters)

        return x, edge_index, edge_attr
