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

from Encoder import Encoder
from Decoder import Decoder
from Seq2Seq import Seq2Seq

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

BATCH_SIZE=256

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device=device
)
#此处，完成数据的准备


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256  #隐藏层向量长度
LAYER1_HID_DIM = 100 #multi-layer embedding
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

print(f'The encoder part has {count_parameters(enc):,} trainable parameters')

print(f'The decoder part has {count_parameters(dec):,} trainable parameters')

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

def initialize_weight(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
if os.path.exists('transformer-luke-model-v1.pt'):
    model.load_state_dict(torch.load('transformer-luke-model.pt'))
else:
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
        
        #trg舍弃最后一位eos
        output, _ = model(src, trg[:,:-1])
        #output [batch_size, trg_len-1, output_dim]
        
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        #[batch*(trg_len-1), output_dim]
        trg = trg[:,1:].contiguous().view(-1)
        #[batch_size * (trg_len-1)]
        
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

N_EPOCHS=30
CLIP=1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    test_loss = evaluate(model, test_iterator, criterion)
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer-luke-model.pt')
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}')
    print(f'\tTest Loss: {test_loss:.3f} | Valid PPL: {math.exp(test_loss):7.3f}')
