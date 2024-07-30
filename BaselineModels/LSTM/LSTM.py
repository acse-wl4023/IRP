import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forget_gate = nn.Sequential(nn.Linear(self.input_size+self.hidden_size, self.hidden_size),
                                         nn.Sigmoid())
        
        self.input_gate = nn.Sequential(nn.Linear(self.input_size+self.hidden_size, self.hidden_size),
                                        nn.Sigmoid())
        
        self.output_gate = nn.Sequential(nn.Linear(self.input_size+self.hidden_size, self.hidden_size),
                                         nn.Sigmoid())
        
        self.candidate_memory_status = nn.Sequential(nn.Linear(self.input_size+self.hidden_size, self.hidden_size),
                                                     nn.Tanh())

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat((x, h_cur), axis=1)
        input = self.input_gate(combined)
        forget = self.forget_gate(combined)
        output = self.output_gate(combined)

        candidate_c = self.candidate_memory_status(combined)

        c_next = forget*c_cur+input*candidate_c
        h_next = output*nn.functional.tanh(c_next)

        return h_next, c_next


class LSTM_encoder(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size):
        super(LSTM_encoder, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.LSTMCell = LSTMCell(self.input_size, self.hidden_size)

    def init_hidden_state(self, x):
        return torch.zeros((x[:, 0, :].shape[0], self.hidden_size), device=x.device),\
               torch.zeros((x[:, 0, :].shape[0], self.hidden_size), device=x.device)
    def forward(self, x):
        # x should be (batch_size, seq_len, num_feature)
        h_cur, c_cur = self.init_hidden_state(x)
        # print(h_cur.shape)
        for i in range(self.seq_len):
            h_cur, c_cur = self.LSTMCell(x[:, i, :], h_cur, c_cur)
        
        return h_cur, c_cur

class LSTM_decoder(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size):
        super(LSTM_decoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.LSTMCell = LSTMCell(self.hidden_size, self.output_size)
        self.linear_h = nn.Linear(self.hidden_size, self.output_size)
        self.linear_c = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, c):
        h_cur = self.linear_h(x)
        c_cur = self.linear_c(c)

        output_list = []
        for i in range(self.seq_len):
            h_cur, c_cur = self.LSTMCell(x, h_cur, c_cur)
            output_list.append(h_cur)

        output = torch.stack(output_list, dim=1)
        return output

class LSTM_Seq2seq(nn.Module):
    def __init__(self, input_seq_len, output_seq_len, input_size, hidden_size, output_size):
        super(LSTM_Seq2seq, self).__init__()
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = LSTM_encoder(self.input_seq_len, self.input_size, self.hidden_size)
        self.decoder = LSTM_decoder(self.output_seq_len, self.hidden_size, self.output_size)

    def forward(self, x):
        h, c = self.encoder(x)
        output = self.decoder(h, c)

        return output