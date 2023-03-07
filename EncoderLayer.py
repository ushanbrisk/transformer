import torch
import torch.nn as nn

from MultiHeadAttentionLayer import MultiHeadAttentionLayer
from PositionwiseFeedForwardLayer import PositionwiseFeedForwardLayer

# 多头注意力实现
class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src: [batch_size, src len, hid_dim]
        # src_mask: [batch_size, 1, 1, src_len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # ff
        _src = self.positionwise_feedforward(src)

        src = self.ff_layer_norm(src + self.dropout(_src))

        return src
