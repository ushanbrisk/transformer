import torch
import torch.nn as nn

from MultiHeadAttentionLayer import MultiHeadAttentionLayer
from PositionwiseFeedForwardLayer import PositionwiseFeedForwardLayer

# #解码层
# 有两个attention层，一个是自attention, 一个是和encoder
# 自注意力，就是让trg作为qkv，用到了trg_mask,
# 和encoder，encoder的编码作为k,v, 而trg的作为q, 其中src_mask表明是否是pad
#
class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)  # 自注意力
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)  # 互注意力
        self.ff_layer_norm = nn.LayerNorm(hid_dim)  # feed forward网络

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)  # 这里用src_mask, 因为k,v都是src

        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.positionwise_feedforward(trg)

        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention
