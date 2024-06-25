from GConvLSTM import GConvLSTMCell
from torchinfo import summary
import Get_data as Gd
import os
import numpy as np

# # Example setup for inputs
# num_nodes = 97149
# num_node_features = 5
# num_classes = 3
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)  # Sample connectivity
# x = torch.randn((num_nodes, num_node_features), dtype=torch.float)
# h = torch.zeros((num_nodes, num_node_features), dtype=torch.float)
# c = torch.zeros((num_nodes, num_node_features), dtype=torch.float)

# # Create an instance of the model
# model = GConvLSTMCell(num_node_features, 12)

# # Use torchinfo to summarize the model
# # Pass inputs as a tuple since your forward method expects two arguments
# summary(model, input_data=(x, edge_index, h, c))

directory = '/scratch_dgxl/sc4623/wl4023/Sibo_22Mar2024'
folders = [os.path.join(directory, f, 'hessian_') for f in os.listdir(directory) if f.startswith('case_')]
files = [os.path.join(folders[0], f) for f in os.listdir(folders[0]) if
                 f.startswith('foward_soln_timestep_') and f.endswith('.npy')]

nodes = Gd.read_data(files)
print(folders[0], np.max(nodes, axis=(0, 1)))

# data_list = []
# for sim in range(30):
#     for steps in range(1,100):
#         try:
#             x = np.load('/scratch_dgxl/sc4623/wl4023/Sibo_22Mar2024/case_0/hessian_/foward_soln_timestep_'+'{0:g}'
#                         .format(float("{:.2f}".format(steps*0.02)))+'.npy')
#             print(x.shape, steps)
#             data_list.append(x)
#         except:
#             pass

# print(np.array(data_list).shape)