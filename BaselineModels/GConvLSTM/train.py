from GConvLSTM import GConvLSTMCell
from torchinfo import summary
import Get_data as Gd
import os
import numpy as np

directory = '/scratch_dgxl/sc4623/wl4023/Sibo_22Mar2024'
folders = [os.path.join(directory, f, 'hessian_') for f in os.listdir(directory) if f.startswith('case_')]
files = [os.path.join(folders[0], f) for f in os.listdir(folders[0]) if
                 f.startswith('foward_soln_timestep_') and f.endswith('.npy')]

nodes = Gd.read_data(files)
print(folders[0], np.max(nodes, axis=(0, 1)))