# from GConvLSTM import GConvLSTMCell
# from torchinfo import summary
# import torch
# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F

# class GCN(torch.nn.Module):
#     def __init__(self, num_node_features, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, 32)
#         self.conv2 = GCNConv(32, num_classes)
#         self.norm = torch.nn.BatchNorm1d(32)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.norm(x)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x

# # Example setup for inputs
# num_nodes = 10
# num_node_features = 5
# num_classes = 3
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)  # Sample connectivity
# x = torch.randn((num_nodes, num_node_features), dtype=torch.float)

# # Create an instance of the model
# model = GCN(num_node_features, num_classes)

# # Use torchinfo to summarize the model
# # Pass inputs as a tuple since your forward method expects two arguments
# summary(model, input_data=(x, edge_index))

print("test")