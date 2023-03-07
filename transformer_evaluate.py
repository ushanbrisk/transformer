
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


from transformer_scratch import evaluate, Seq2Seq, Encoder, Decoder

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256  #隐藏层向量长度
ENC_LAYERS = 3 #编码器堆叠层数
DEC_LAYERS = 3  #解码器堆叠层数
ENC_HEADS = 8   #多头数目
DEC_HEADS = 8   #多头数目
ENC_PF_DIM = 512 #编码器前馈网络中间隐藏向量长度
DEC_PF_DIM = 512 #解码器前馈网络中间隐藏向量长度
ENC_DROPOUT = 0.1 #编码器dropout率
DEC_DROPOUT = 0.1 #解码器dropout率


enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)


model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)#模型转到cuda上



