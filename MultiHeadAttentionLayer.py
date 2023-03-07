import torch
import torch.nn as nn

# #多头注意力层
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim  # 向量长度　总长
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads  # 每个头的向量长度

        self.fc_q = nn.Linear(hid_dim, hid_dim)  # hid_dim -> hid_dim向量映射
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)  # 这里共有HxHx4个参数

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)  # cuda

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query: [batch_size, query_len, hid dim]
        # key  : [batch_size,   key_len, hid dim]
        # value: [batch_size, value_len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # [batch_size, n_heads, src_len, head_dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # [batch_size, n_heads, q_len, k_len]

        if mask is not None:
            energy.masked_fill(mask == 0, -1e10)  # 注意这里的mask的shape　这一步能否提前?　[batch_size, 1, 1, src_len]

        attention = torch.softmax(energy, dim=-1)  # [batch_size, n_heads, q_len, k_len]
        # 这里的attentio本质就是一个归一化的权重，代表里q对k,v的
        # 共有head个通道，

        x = torch.matmul(self.dropout(attention), V)  # 这里用了dropout,想不到
        # batch_size, n_heads, q_len, head_dim

        x = x.permute(0, 2, 1, 3).contiguous()  # 深度拷贝, 不影响原来

        x = x.view(batch_size, -1, self.hid_dim)
        # [batch_size, query_len, hid_dim]

        x = self.fc_o(x)  # [batch_size, query_len, hid_dim]

        return x, attention

