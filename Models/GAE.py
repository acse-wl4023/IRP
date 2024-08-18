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

class CustomGCNConv(GCNConv):
    def forward(self, x, edge_index, edge_attr):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf detected in input to CustomGCNConv")
        out = super().forward(x, edge_index, edge_attr)
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            weight = self.lin.weight.data
            bias = self.lin.bias.data if self.lin.bias is not None else None
            print(weight)
            print(bias)
            print(x)
            print(edge_index)
            print(edge_attr)
            print(out)
            print("NaN or Inf detected in output of CustomGCNConv")
            
        # 检查中间结果
        with torch.no_grad():
            weight = self.lin.weight.data
            if torch.isnan(weight).any() or torch.isinf(weight).any():
                print("NaN or Inf detected in GCNConv weight")

            bias = self.lin.bias.data if self.lin.bias is not None else None
            if bias is not None and (torch.isnan(bias).any() or torch.isinf(bias).any()):
                print("NaN or Inf detected in GCNConv bias")
        return out


# Edge pooling
def pool_edge_mean(cluster, edge_index, edge_attr):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1) 
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='mean')
    return edge_index, edge_attr


def avg_pool_mod_no_x(cluster, edge_index, edge_attr):
    cluster, perm = consecutive_cluster(cluster)
    # Pool edge attributes 
    edge_index_pool, edge_attr_pool = pool_edge_mean(cluster, edge_index, edge_attr)
    # Pool node positions 
    # pos_pool = None if pos is None else scatter_mean(pos, cluster, dim=0)
    return edge_index_pool, edge_attr_pool, cluster, perm


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
                 num_mp_layers: List[int], 
                 clusters: List[int],
                 centroids,
                 hidden_channels: int,
                 n_mlp_mp: int):
        super(Encoder, self).__init__()
        self.input_node_channel = input_node_channel
        self.hidden_channels = hidden_channels
        self.edge_aggregator = EdgeAggregator() # 精细图向粗图汇聚
        self.clusters = clusters
        self.centroids = centroids

        # number of message passing layers in each blocks
        self.num_mp_layers = num_mp_layers

        # depth of the encoder
        self.depth = len(self.num_mp_layers) # depth-1次聚类
        if self.depth != len(self.clusters)+1:
            assert self.depth == self.num_clusters + 1, "Depth must be equal to num_clusters + 1" 

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

        # f2c时，精细图与粗图需要建立连接
        # 用于生成f2c时的边的特征
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
        for j in range(self.n_mlp_mp): # 任意数量的linear都可以，只是这里用n_mlp_mp
            if j == 0: 
                input_features = self.input_node_channel
                output_features = self.hidden_channels
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.node_embeding_mlp.append(nn.Linear(input_features, output_features))

        # 利用linear层，根据精细图的节点值和粗细图连接的特征，生成f2c的边特征，
        # 之后进行edge_aggregate
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
        

        # 对graph的边特征进行一个编码的操作
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
        # 对于每张初始图，边的特征都是1，这不利于后续数据的压缩
        # 用edge_encoder_mlp进行重新编码

        # node feature embedding
        for j in range(self.n_mlp_mp):
            x = self.node_embeding_mlp[j](x)
            x = self.act(x)

        if torch.isnan(x).any():
            print("NaN detected in node embedding MLP")

        # # 第一层  边特征非线性变换然后进行message passing
        # edge_attr = edge_attr.fill_(1)
        # use mlp update the attr of edge of coarse graph
        for j in range(self.n_mlp_mp):
            edge_attr = self.edge_encoder_mlp[0][j](edge_attr)
            edge_attr = self.act(edge_attr)
            if torch.isnan(edge_attr).any():
                print(f"NaN detected in edge encoder MLP layer {j}")

        # print(edge_attr.shape, edge_index.shape)
        for j in range(self.num_mp_layers[0]):
            x = self.mp_blocks[0][j](x, edge_index, edge_attr)
            x = self.act(x)
            if torch.isnan(x).any():
                print(f"NaN detected in encoder message passing layer {j} of block 0")

        # 储存f2c的边
        edge_indices_f2c = []
        edge_indices = [edge_index]
        node_attrs = [x]
        edge_attrs = [edge_attr]
        position = [pos] # 存储生成的每张图的节点的坐标 从最精细到最粗糙
        if torch.isnan(x).any():
            print("NaN detected in encoder:", torch.isnan(x).any(), x.max(), x.min())

        for i in range(1, self.depth):
            # 聚类
            pos_fine = torch.clone(pos)
            # cluster, pos = kmeans(pos, self.num_clusters[i-1], device=pos.device)
            cluster = self.clusters[i-1].to(x.device)
            pos = self.centroids[i-1].to(x.device)

            # 计算得到新的图的边
            edge_index, edge_attr, cluster, perm = avg_pool_mod_no_x(cluster,
                                                                    edge_index,
                                                                    edge_attr)

            # edge_attr.fill_(1)
            for j in range(self.n_mlp_mp):
                edge_attr = self.edge_encoder_mlp[i][j](edge_attr)
                edge_attr = self.act(edge_attr)
                if torch.isnan(edge_attr).any():
                    print(f"NaN detected in edge encoder MLP layer {j} of block {i}")

            position.append(pos)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)

            n_nodes = x.shape[0]
            edge_index_f2c = torch.concat( (torch.arange(0, n_nodes, dtype=torch.long, device=x.device).view(1,-1), cluster.view(1,-1)), axis=0 )
            edge_indices_f2c.append(edge_index_f2c)
            pos_coarse = torch.clone(pos)


            # 在fine graph和coarsen graph间构建连接，用于计算coarsen graph节点的特征值
            # coarsen graph的节点是cluster的中点，利用距离来计算权重
            # print(pos_coarse.device, edge_index_f2c.device, pos_fine.device)
            edge_attr_f2c = pos_coarse[edge_index_f2c[1,:]] - pos_fine[edge_index_f2c[0,:]]
            edge_attr_f2c = edge_attr_f2c / (torch.norm(edge_attr_f2c, dim=-1, keepdim=True) + 1e-8)

            # encode the f2c edge attributes with MLP
            for j in range(self.n_mlp_mp):
                # print(edge_attr_f2c.dtype)
                edge_attr_f2c = self.edge_encoder_f2c_mlp[j](edge_attr_f2c)
                edge_attr_f2c = self.act(edge_attr_f2c)
                if torch.isnan(edge_attr_f2c).any():
                    print(f"NaN detected in edge encoder F2C MLP layer {j} of block {i}")

            temp_ea = torch.cat((edge_attr_f2c, x), axis=1)
            for j in range(self.n_mlp_mp):
                temp_ea = self.downsample_mlp[j](temp_ea)
                temp_ea = self.act(temp_ea)
            temp_ea = edge_attr_f2c + temp_ea
            temp_ea = self.downsample_norm(temp_ea)
            x = self.edge_aggregator((pos_fine, pos_coarse), edge_index_f2c, temp_ea)

            if torch.isnan(x).any():
                print(f"NaN detected after edge aggregation in block {i}")


            for j in range(self.num_mp_layers[i]):
                x = self.mp_blocks[i][j](x, edge_index, edge_attr)
                if torch.isnan(x).any():
                    print(f"NaN detected in input x in message passing layer before activate {j} of block {i}")
                    # print(f"edge_index: {edge_index}")
                    # print(f"edge_attr: {edge_attr}")

                if torch.isnan(edge_index).any():
                    print(f"NaN detected in edge_index in message passing layer before activate {j} of block {i}")
                if torch.isnan(edge_attr).any():
                    print(f"NaN detected in edge_attr in message passing layer before activate {j} of block {i}")
                x = self.act(x)
                if torch.isnan(x).any():
                    print(f"NaN detected in encoder message passing layer {j} of block {i}")

            node_attrs.append(x)

        return x, edge_index, edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, self.clusters



class Decoder(nn.Module):
    def __init__(self, 
                 output_node_channel: int,
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
            
        self.conv = nn.Conv1d(1, 1, 3, 1, 1)
        # self.coarsen = nn.ModuleList()
        # self.coarsen.append(nn.Linear(100, 97149))
        # self.coarsen.append(nn.Linear(5, 100))
        
        # self.coarsen_gcn = nn.ModuleList()
        # self.coarsen_gcn.append(CustomGCNConv(self.hidden_channels, self.hidden_channels))
        # self.coarsen_gcn.append(CustomGCNConv(self.hidden_channels, self.hidden_channels))
        

    def forward(self, x, edge_index, edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters):

        # 解码器第一层
        # use mlp update the attr of edge of coarse graph
        for j in range(self.n_mlp_mp):
            edge_attr = self.edge_decoder_mlp[0][j](edge_attr)
            edge_attr = self.act(edge_attr)
            if torch.isnan(edge_attr).any():
                print(f"NaN detected in edge decoder MLP layer {j} of block 0")

        # 第一层的消息传递
        for j in range(self.num_mp_layers[0]):
            x = self.mp_blocks[0][j](x, edge_index, edge_attr)
            x = self.act(x)
            if torch.isnan(x).any():
                print(f"NaN detected in decoder message passing layer {j} of block 0")

        for i in range(self.depth-1, 0, -1):
            """
            图细化
            """
            # 分别获取粗细图的节点坐标和连接他们的边的属性
            pos_fine = position[i-1]
            pos_coarse = position[i]
            edge_index_c2f = edge_indices_f2c[i-1][[1, 0]] # 上下交换

            edge_attr_c2f = pos_coarse[edge_index_c2f[0,:]] - pos_fine[edge_index_c2f[1,:]]
            # encode the edge attributes with MLP
            for j in range(self.n_mlp_mp):
                edge_attr_c2f = self.edge_decoder_c2f_mlp[j](edge_attr_c2f)
                edge_attr_c2f = self.act(edge_attr_c2f)
                if torch.isnan(edge_attr_c2f).any():
                    print(f"NaN detected in edge decoder C2F MLP layer {j} of block {i}")

            # 插值法将图初步细化
            x = x[clusters[i-1]]
            # x = x.permute(1, 0)
            # # print(x.shape)
            # x = self.coarsen[i-1](x)
            # x = x.permute(1, 0)
            # x = self.coarsen_gcn()
            # x = self.act(x)
            
            
            # 得到c2f的边的特征
            temp_ea = torch.cat((edge_attr_c2f, x), axis=1)

            for j in range(self.n_mlp_mp):
                temp_ea = self.upsample_mlp[j](temp_ea)
                temp_ea = self.act(temp_ea)
            temp_ea = edge_attr_c2f + temp_ea

            temp_ea = self.upsample_norm(temp_ea)

            # 通过消息传递得到精细图的节点特征
            x = self.edge_aggregator((pos_coarse, pos_fine), edge_index_c2f, temp_ea)

            if torch.isnan(x).any():
                print(f"NaN detected after edge aggregation in block {i}")

            # 更新精细图的边索引及其特征
            edge_index = edge_indices[i-1]
            edge_attr = edge_attrs[i-1]
            for j in range(self.n_mlp_mp):
                edge_attr = self.edge_decoder_mlp[i][j](edge_attr)
                edge_attr = self.act(edge_attr)
                if torch.isnan(edge_attr).any():
                    print(f"NaN detected in edge decoder MLP layer {j} of block {i}")

            for j in range(self.num_mp_layers[i]):
                x = self.mp_blocks[i][j](x, edge_index, edge_attr)
                x = self.act(x)
                if torch.isnan(x).any():
                    print(f"NaN detected decoder in message passing layer {j} of block {i}")
        
        for j in range(self.n_mlp_mp):
            x = self.node_decoder_mlp[j](x)
            x = self.act(x)
        if torch.isnan(x).any():
            print(f"NaN detected in final layer {j} of block {i}")
        x = x.unsqueeze(0).permute(0, 2, 1)
        x = self.conv(x)
        x = x.squeeze(0).permute(1, 0)
        
    

        return x, edge_index, edge_attr


class GAE(nn.Module):
    def __init__(self, 
                 input_node_channel: int,
                 output_node_channel: int,
                 num_mp_layers: List[int],
                 clusters: List[int],
                 centroids,
                 hidden_channels: int,
                 n_mlp_mp: int):
        super(GAE, self).__init__()
        self.input_node_channel = input_node_channel
        self.output_node_channel = output_node_channel
        self.num_mp_layers = num_mp_layers
        self.clusters = clusters
        self.centroids = centroids
        self.hidden_channels = hidden_channels
        self.n_mlp_mp = n_mlp_mp

        self.encoder = Encoder(self.input_node_channel,
                               self.num_mp_layers,
                               self.clusters,
                               self.centroids,
                               self.hidden_channels,
                               self.n_mlp_mp)
        
        self.decoder = Decoder(self.output_node_channel,
                               self.num_mp_layers,
                               self.hidden_channels,
                               self.n_mlp_mp)

    def forward(self, x, edge_index, edge_attr, pos):
        x, edge_index, edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters = self.encoder(x, edge_index, edge_attr, pos)
        # print(x.max(), x.min())
        x, edge_index, edge_attr = self.decoder(x, edge_index, edge_attr, edge_indices, edge_attrs, edge_indices_f2c, position, node_attrs, clusters)

        return x, edge_index, edge_attr