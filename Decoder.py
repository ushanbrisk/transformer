import torch
import torch.nn as nn
from MultiLayerEmbedding import MultiLayerEmbedding
from DecoderLayer import DecoderLayer

# #解码器
# 编码器得到向量序列，
# 解码器与编码器类似，也是多个层重复多次，但是每个层内有两个多头层，一个多头层是有mask的，对target做一次，作为query, 然后用encoder的输出作为key, value, 再做一次多头注意力
#
# 位置编码+token嵌入编码, 位置编码最长100
#
class Decoder(nn.Module):
    def __init__(self,
                 output_dim,  # target的字库大小
                 hid_dim,  # hidden vector length
                 layer1_hid_dim,  # 第一层隐向量长度
                 n_layers,  # 层数
                 n_heads,  # 多头数
                 pf_dim,  # 每个多头注意力曾里面的ff中间数目
                 dropout,
                 device,
                 max_length=100):
        super().__init__()
        self.device = device
        self.tok_embedding = MultiLayerEmbedding(output_dim, hid_dim, layer1_hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)  # 从hidden space映射回字库
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    # src, target,输入，　以及他们各自的mask都要作为输入
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch_size, trg_len]
        # enc_src: [batch_size, src_len, hid_dim]
        # src_mask: [batch_size, 1, 1, src_len]
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        # trg_mask是特别的，因为考虑了单向性，
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)  # 这是新定义
        # 这里pos直接定义在cuda上面，所以默认传入的数据，也都是在cuda上面，才好做运算
        # [batch_size, trg_len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg: [batch_size, trg_len, hid_dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg:[batch_size, trg len, hid dim]

        output = self.fc_out(trg)
        # [batch_size, trg_len, output_dim]

        return output, attention