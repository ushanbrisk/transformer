import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
import os

SEED=1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True

#创建tokenize
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize = tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

TRG = Field(tokenize = tokenize_en,
           init_token = '<sos>',
           eos_token = '<eos>',
           lower = True,
           batch_first = True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de','.en'), fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE=128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device=device
)
#此处，完成数据的准备


# 编码器
# #输入经过标准嵌入层，再加上位置嵌入，位置共有100个词汇,原始论文中没有位置嵌入，而是用固定三角函数，最新的，包括bert都是用位置嵌入；这里面用
# elementwise的加法，合理吗?
# 
# 做加法的时候，需要加上尺度因子，sqrt(hid_dim)，据说能够减少方差，让模型容易训练，这个理论证明呢?
# 
# src_mask: 形状与source sentence一致，如果token是pad, 为0; 如果token 不是pad, 为１
# 

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

    #src shape:[batch_size, src_len]
    def forward(self, src):
        
        x = self.emb1(src) 
        #x shape:[batch_size, src_len, layer1_dim]
        
        x = self.emb2(x)
        #x shape:[batch_size, src_len, hid_dim]
        
        return x
    #返回 [batch_size, src_len, hid_dim]
        
class Encoder(nn.Module):
    def __init__(self,
                input_dim,   #文本字库大小
                hid_dim,  #隐向量长度
                layer1_hid_dim, #第一层隐向量长度
                n_layers, #层数
                n_heads, #head数
                pf_dim, #
                dropout, #
                device, #
                max_length = 100): #输入句子最大长度)
        super().__init__()
        
        self.device = device
        self.tok_embedding = MultiLayerEmbedding(input_dim, hid_dim, layer1_hid_dim) #从字库做嵌入
        self.pos_embedding = nn.Embedding(max_length, hid_dim) #位置嵌入
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)   
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout) #层实现
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device) #传入cuda
        
        #需要提供src_mask
    def forward(self, src, src_mask):
        #src: [batch_size, src_len]
        #src_mask: [batch_size, 1, 1, src_len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device) #传入cuda
        #pos:[batch_size, src_len]
        
        src = self.dropout( (self.tok_embedding(src)*self.scale) + self.pos_embedding(pos))
        #[batch_size, src_len, hid_dim]
        
        for layer in self.layers:
            src = layer(src, src_mask) #每个encoderlayer输入都用到同样的src_mask
            
        #src shape: [batch_size, src_len_hid_dim]
        return src

# 前馈网络层，这个才是真正的神经网络，前面是特征提取
class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)#共有H x pf_dim x 2个参数
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #x [batch_size, query_len, hid_dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        #[batch_size, query_len, hid_dim]
        
        return x

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
        #src: [batch_size, src len, hid_dim]
        #src_mask: [batch_size, 1, 1, src_len]
        
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #ff 
        _src = self.positionwise_feedforward(src)
        
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src

# #多头注意力层
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim%n_heads==0
        
        self.hid_dim = hid_dim #向量长度　总长
        self.n_heads = n_heads
        self.head_dim = hid_dim//n_heads #每个头的向量长度
        
        
        #在手动实现lstm的代码中，可以讲q,k,v放到一个大矩阵里面,可以参考一下
        
        self.fc_q = nn.Linear(hid_dim, hid_dim) #hid_dim -> hid_dim向量映射
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)#这里共有HxHx4个参数
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device) #cuda
        
    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]
        
        #query: [batch_size, query_len, hid dim]
        #key  : [batch_size,   key_len, hid dim]
        #value: [batch_size, value_len, hid dim]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #[batch_size, n_heads, src_len, head_dim]
        
        energy = torch.matmul(Q, K.permute(0,1,3,2))/self.scale
        #[batch_size, n_heads, q_len, k_len]
        
        if mask is not None:
            energy.masked_fill(mask == 0, -1e10) #注意这里的mask的shape
        
        attention = torch.softmax(energy, dim=-1) #[batch_size, n_heads, q_len, k_len]
        
        x = torch.matmul(self.dropout(attention), V) #这里用了dropout,想不到
        #batch_size, n_heads, q_len, head_dim
        
        x = x.permute(0, 2, 1, 3).contiguous() #深度拷贝, 不影响原来
        
        x = x.view(batch_size, -1, self.hid_dim)
        #[batch_size, query_len, hid_dim]
        
        x = self.fc_o(x) #[batch_size, query_len, hid_dim]
        
        return x, attention

# #解码器
# 编码器得到向量序列，
# 解码器与编码器类似，也是多个层重复多次，但是每个层内有两个多头层，一个多头层是有mask的，对target做一次，作为query, 然后用encoder的输出作为key, value, 再做一次多头注意力
# 
# 位置编码+token嵌入编码, 位置编码最长100
# 
class Decoder(nn.Module):
    def __init__(self,
                output_dim, #target的字库大小
                hid_dim, #hidden vector length
                layer1_hid_dim, #第一层隐藏向量大小
                n_layers, #层数
                n_heads, #多头数
                pf_dim, #每个多头注意力曾里面的ff中间数目
                dropout, 
                device,
                max_length = 100):
        
        super().__init__()
        self.device = device
        self.tok_embedding = MultiLayerEmbedding(output_dim, hid_dim, layer1_hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([ DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim) #从hidden space映射回字库
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    #src, target,输入，　以及他们各自的mask都要作为输入    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        #trg: [batch_size, trg_len]
        #enc_src: [batch_size, src_len, hid_dim]
        #src_mask: [batch_size, 1, 1, src_len]
        #trg_mask: [batch_size, 1, trg_len, trg_len]
        #trg_mask是特别的，因为考虑了单向性，
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device) #这是新定义
        #这里pos直接定义在cuda上面，所以默认传入的数据，也都是在cuda上面，才好做运算
        #[batch_size, trg_len]
        
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        #trg: [batch_size, trg_len, hid_dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg:[batch_size, trg len, hid dim]
        
        output = self.fc_out(trg)
        #[batch_size, trg_len, output_dim]
        
        return output, attention


# #解码层
# 有两个attention层，一个是自attention, 一个是和encoder
# 自注意力，就是让trg作为qkv，用到了trg_mask,
# 和encoder，encoder的编码作为k,v, 而trg的作为q, 其中src_mask表明是否是pad
class DecoderLayer(nn.Module):
    def __init__(self,
                hid_dim,
                n_heads,
                pf_dim,
                dropout,
                device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim) #自注意力
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim) #互注意力
        self.ff_layer_norm = nn.LayerNorm(hid_dim) #feed forward网络
        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(hid_dim, pf_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask) #这里用src_mask, 因为k,v都是src
        
        trg = self.enc_attn_layer_norm( trg + self.dropout(_trg))
        
        _trg = self.positionwise_feedforward(trg)
        
        trg = self.ff_layer_norm( trg + self.dropout(_trg))
        
        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self,
                encoder,
                decoder,
                src_pad_idx,
                trg_pad_idx,
                device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx #pad在输入字库中的编号
        self.trg_pad_idx = trg_pad_idx #pad在输出字库中的编号
        self.device = device
        
    #制作src的模板
    def make_src_mask(self, src):
        #src  [batch_size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) 
        #[batch size ,1 ,1 , src_len]  pad为0, 不是pad则为１
        
        return src_mask
    
    #制作trg的模板
    def make_trg_mask(self, trg):
        #trg:[batch_size, trg_len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) #bool类型
        #[batch_size, 1, 1, trg_len]
        
        #输出的长度
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()#直接定义在cuda上?
        #shape: [trg_len, trg_len]
        
        trg_mask = trg_pad_mask & trg_sub_mask
        #[batch_size, 1, trg_len, trg_len]
        
        return trg_mask
    
    def forward(self, src, trg):
        #[batch_size, src_len]
        #[batch_size, trg_len]
        
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        
        #
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output:[batch_size, trg_len, output-dim]
        #attention:[batch_size, n_heads, trg_len, src_len]
        
        return output, attention

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256  #隐藏层向量长度
LAYER1_HID_DIM = 100 #第一层隐藏嵌入向量长度
ENC_LAYERS = 3 #编码器堆叠层数
DEC_LAYERS = 3  #解码器堆叠层数
ENC_HEADS = 8   #多头数目
DEC_HEADS = 8   #多头数目
ENC_PF_DIM = 512 #编码器前馈网络中间隐藏向量长度
DEC_PF_DIM = 512 #解码器前馈网络中间隐藏向量长度
ENC_DROPOUT = 0.1 #编码器dropout率
DEC_DROPOUT = 0.1 #解码器dropout率

#编码器参数多层部分: (HID_DIM*HID_DIM*4 + HID_DIM*ENC_PF_DIM*2)*ENC_LAYERS，　
#    嵌入层输入：　　INPUT_DIM x HID_DIM    位置嵌入: max_length * HID_DIM
#    输出:不需要

#解码器参数: 输入　OUTPUT_DIM*HID_DIM + max_length*HID_DIM
#解码器参数，多层部分: HID_DIM*HID_DIM*3, 
#多头的数目，不影响参数量
enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              LAYER1_HID_DIM,
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              LAYER1_HID_DIM,
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)#模型转到cuda上

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

print(f'The model has {count_parameters(enc):,} trainable parameters')

#参数统计
# 多头注意力层: HID_DIM*(HID_DIM+1)*4
# norm层：HID_DIM + HID_DIM
# FF层: HID_DIM*(ENC_PF_DIM+1)+(HID_DIM+1)*ENC_PF_DIM
# norm层次:HID_DIM + HID_DIM
#
# 输入嵌入:INPUT_DIM * HID_DIM
# 位置嵌入:100*HID_DIM
# 打印各个参数代码
for name, param in enc.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)

each=(HID_DIM*(HID_DIM+1)*4 + HID_DIM + HID_DIM + HID_DIM*(ENC_PF_DIM+1)+(HID_DIM+1)*ENC_PF_DIM+HID_DIM + HID_DIM)
enc_total = each*ENC_LAYERS + (INPUT_DIM + HID_DIM)*LAYER1_HID_DIM + 100*HID_DIM
assert enc_total==count_parameters(enc)

# 输入 OUTPUT_DIM * HID_DIM
# 输出 (HID_DIM + 1)* OUTPUT_DIM
# 位置 100 * HID_DIM
# 每层参数
#     self_attention:HID_DIM * (HID_DIM+1) * 4
#     self_attn_layer_norm:HID_DIM + HID_DIM
#     encoder_attention:HID_DIM * (HID_DIM+1) * 4
#     enc_attn_layer_norm:HID_DIM + HID_DIM
#     positionwise_feedforward: (HID_DIM+1)*DEC_PF_DIM + HID_DIM * (DEC_PF_DIM + 1)
#     ff_layer_norm: HID_DIM + HID_DIM
# 总共参数
#     each = HID_DIM * (HID_DIM+1) * 4 + HID_DIM + HID_DIM + HID_DIM * (HID_DIM+1) * 4 + HID_DIM + HID_DIM +  (HID_DIM+1)*DEC_PF_DIM + HID_DIM * (DEC_PF_DIM + 1) + HID_DIM + HID_DIM
#     OUTPUT_DIM * HID_DIM+(HID_DIM + 1)* OUTPUT_DIM+100 * HID_DIM +
#

each = HID_DIM * (HID_DIM+1) * 4 + HID_DIM + HID_DIM + HID_DIM * (HID_DIM+1) * 4 + HID_DIM + HID_DIM +  (HID_DIM+1)*DEC_PF_DIM + HID_DIM * (DEC_PF_DIM + 1) + HID_DIM + HID_DIM
dec_total = each * DEC_LAYERS + (OUTPUT_DIM + HID_DIM)*LAYER1_HID_DIM+(HID_DIM + 1)* OUTPUT_DIM+100 * HID_DIM
assert dec_total==count_parameters(dec)

count_parameters(model) * 2/1024/1024

def initialize_weight(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform(m.weight.data)

model.apply(initialize_weight)

LEARNING_RATE=0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
#这里包括了softmax, log的操作, trg序列，输入只考虑 trg[:-1], 输出只用pred[1:],不包括第一位的sos

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0

    #一个epoch的loss,也就是所有数据过一遍
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        #trg: [ batch_size, trg_len ]
        
        optimizer.zero_grad()
        
        #trg舍弃最后一位
        output, _ = model(src, trg[:,:-1])
        #output [batch_size, trg_len-1, output_dim]
        
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        #[batch*(trg_len-1), output_dim]
        trg = trg[:,1:].contiguous().view(-1)
        #[batch_size, trg_len-1]
        
        loss = criterion(output, trg)
        
        loss.backward() #各个梯度都是记录在parameter上面
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step() #因为optimizer已经针对model.parameters()了，所以这一步就是让model更新参数
        
        epoch_loss += loss.item()
    return epoch_loss/len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output=output.contiguous().view(-1, output_dim)
            trg=trg[:,1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss/len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS=10
CLIP=1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer-luke-model.pt')
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}')

