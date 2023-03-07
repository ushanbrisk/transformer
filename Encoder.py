import torch
import torch.nn as nn

from EncoderLayer import EncoderLayer
from MultiLayerEmbedding import MultiLayerEmbedding

# 编码器
# #输入经过标准嵌入层，再加上位置嵌入，位置共有100个词汇,原始论文中没有位置嵌入，而是用固定三角函数，最新的，包括bert都是用位置嵌入；这里面用
# elementwise的加法，合理吗?
#
# 做加法的时候，需要加上尺度因子，sqrt(hid_dim)，据说能够减少方差，让模型容易训练，这个理论证明呢?
#
# src_mask: 形状与source sentence一致，如果token是pad, 为0; 如果token 不是pad, 为１
#


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,  # 文本字库大小
                 hid_dim,  # 隐向量长度
                 layer1_hid_dim,  # 第一层隐向量长度
                 n_layers,  # 层数
                 n_heads,  # head数
                 pf_dim,  #
                 dropout,  #
                 device,  #
                 max_length=100):  # 输入句子最大长度)
        super().__init__()

        self.device = device
        self.tok_embedding = MultiLayerEmbedding(input_dim, hid_dim, layer1_hid_dim)  # 从字库做嵌入
        self.pos_embedding = nn.Embedding(max_length, hid_dim)  # 位置嵌入

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)  # 层实现
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)  # 传入cuda

        # 需要提供src_mask

    def forward(self, src, src_mask):
        # src: [batch_size, src_len]
        # src_mask: [batch_size, 1, 1, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)  # 传入cuda
        # pos:[batch_size, src_len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # [batch_size, src_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)  # 每个encoderlayer输入都用到同样的src_mask

        # src shape: [batch_size, src_len_hid_dim]
        return src
