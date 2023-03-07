import torch
import torch.nn as nn


#here using low rank assumption, decomposing into two embedding matrix multiplication
#the purpose is to reduce the number of parameters
class MultiLayerEmbedding(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 layer1_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.layer1_dim = layer1_dim

        self.emb1 = nn.Embedding(input_dim, layer1_dim)
        self.emb2 = nn.Linear(in_features=layer1_dim, out_features=hid_dim, bias=False)

    # src shape:[batch_size, src_len]
    def forward(self, src):
        x = self.emb1(src)
        # x shape:[batch_size, src_len, layer1_dim]

        x = self.emb2(x)
        # x shape:[batch_size, src_len, hid_dim]

        return x
    # 返回 [batch_size, src_len, hid_dim]
