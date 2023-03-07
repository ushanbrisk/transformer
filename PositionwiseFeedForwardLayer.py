import torch
import torch.nn as nn

# 前馈网络层，这个才是真正的神经网络，前面的multi-head attention是特征提取
class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)  # 共有H x pf_dim x 2个参数
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x [batch_size, query_len, hid_dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        # [batch_size, query_len, hid_dim]

        return x

