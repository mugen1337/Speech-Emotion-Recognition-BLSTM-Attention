import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class lld_blstm_attn(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=32, dropout=0.3):
        super(lld_blstm_attn, self).__init__()
        self.fsl = nn.Linear(input_dim, hidden_dim)
        self.d1 = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.d2 = nn.Dropout(p=dropout)

        # attn
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.attn_softmax = nn.Softmax(dim=1)

        self.d3 = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(hidden_dim * 2, 4)

    def forward(self, x, mask):
        x = F.relu(self.fsl(x))
        x = self.d1(x)

        (x, _) = self.lstm(x)
        x = self.d2(x)

        # attention
        a = self.attn(x)
        a = a.squeeze(dim=2)
        alpha = self.attn_softmax(a * mask).unsqueeze(-1)
        x = x.mul(alpha)
        x = x.sum(dim=1)

        x = self.d3(x)
        x = self.output_layer(x)
        return x