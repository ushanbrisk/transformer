import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx  # pad在输入字库中的编号
        self.trg_pad_idx = trg_pad_idx  # pad在输出字库中的编号
        self.device = device

    # 制作src的模板
    def make_src_mask(self, src):
        # src  [batch_size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # [batch size ,1 ,1 , src_len]  pad为0, 不是pad则为１

        return src_mask

    # 制作trg的模板
    def make_trg_mask(self, trg):
        # trg:[batch_size, trg_len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # bool类型
        # [batch_size, 1, 1, trg_len]

        # 输出的长度
        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()  # 直接定义在cuda上?
        # shape: [trg_len, trg_len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # [batch_size, 1, trg_len, trg_len]

        return trg_mask

    def forward(self, src, trg):
        # [batch_size, src_len]
        # [batch_size, trg_len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)

        #enc_src is the encoder of input sequence
        #
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output:[batch_size, trg_len, output-dim]
        # attention:[batch_size, n_heads, trg_len, src_len]

        return output, attention
