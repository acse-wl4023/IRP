import torch
import torch.nn as nn
from torch.nn import Parameter

class LSTM_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 输入门i_t
        self.W_i = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))
        # 遗忘门f_t
        self.W_f = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))
        # 候选内部状态g_t
        self.W_g = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_g = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size))
        # 输出门o_t
        self.W_o = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


    def forward(self, x_t, h_t, c_t): # input = (batch_size, input_size)

        # 更新门组件及内部候选状态（Tips:Pytorch中@用于矩阵相乘，*用于逐个元素相乘）
        i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
        f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
        g_t = torch.tanh(x_t @ self.W_g + h_t @ self.U_g + self.b_g)
        o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
        # 记忆单元和隐藏单元更新
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
    


