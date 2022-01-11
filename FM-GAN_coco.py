# -*- coding:utf-8 -*-
# %%
import os
import random
import math

import argparse
from tqdm import tqdm

import numpy as np
import _pickle as cPickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from word_correction import tokenizer

from torchtext.legacy import data, datasets
import torchtext
import dill

from generator import Generator
from embedding import Embedding
from feature import Fextractor
from utils import sample_from_generator, IPOT_distance, sample_from_generator_soft,get_minibatches_idx

# %%
# ================== Parameter Definition =================
BATCH_SIZE = 64 * 4
SEQ_LEN = 50
PRE_EPOCH_NUM = 50
SEED = 88

emb_dim = 300
latant_dim = 128
hidden_dim = 256
use_cuda = True

# d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
# d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dropout = 0.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# %%
# ================== Dataset Definition =================
trainpath = "./data/MS_COCO/14/train_coco_14.txt"
testpath = "./data/MS_COCO/14/test_coco_14.txt"
train, val = np.loadtxt(trainpath), np.loadtxt(testpath)
ixtoword, _ = cPickle.load(
    open('./data/MS_COCO/14/vocab_coco_14.pkl', 'rb'))
ixtoword = {i: x for i, x in enumerate(ixtoword)}

VOCAB_SIZE = 27842

train, val = train[:120000], val[:10000]
print('Total words: %d' % VOCAB_SIZE)

###############################################################################
# %%
embedding = Embedding(VOCAB_SIZE, emb_dim)
generator = Generator(latant_dim, VOCAB_SIZE, emb_dim, hidden_dim, use_cuda)
extractor = Fextractor(emb_dim, dropout)

if use_cuda:
    embedding = embedding.cuda()
    generator = generator.cuda()
    extractor = extractor.cuda()
# %%
# pretrained_embeddings = TEXT.vocab.vectors
# print(pretrained_embeddings.shape)
# embedding.emb.weight.data.copy_(pretrained_embeddings)

# unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
# pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
# init_idx = TEXT.vocab.stoi[TEXT.init_token]
# eos_idx = TEXT.vocab.stoi[TEXT.eos_token]

# embedding.emb.weight.data[unk_idx] = torch.zeros(emb_dim)
# embedding.emb.weight.data[pad_idx] = torch.zeros(emb_dim)
# embedding.emb.weight.data[init_idx] = torch.zeros(emb_dim)
# embedding.emb.weight.data[eos_idx] = torch.zeros(emb_dim)

# print(embedding.emb.weight.data)
# %%
# embedding_optim = optim.Adam(embedding.parameters(),lr=0.01)
# generator_optim = optim.Adam(generator.parameters(),lr=0.01)
# %%
bs = 1
z = torch.randn((1, bs, latant_dim)).cuda()
samples = sample_from_generator(generator, embedding,bs,50,z)
# samples = sample_from_generator_soft(generator, embedding,bs,50,z)
print(' '.join([ixtoword[i.item()] for i in samples[0]]))

# %%
# ================== Pretrain with MLE =================
# def train_epoch(model, data_iter, criterion, optimizerG,optimizerE):
#     total_loss = 0.
#     total_words = 0.
#     for data in tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
#         data = Variable(data.text[1:])
#         z = torch.randn((1, data.size(0), latant_dim)).cuda()
#         # z = torch.zeros((1, data.size(0), hidden_dim)).cuda()
#         if use_cuda:
#             data = data.cuda()
#         emb = embedding(data)
#         pred = model.forward(emb, z)
#         loss = criterion(pred[:-1], data.view(-1)[1:])
#         total_loss += loss.item()
#         total_words += data.size(0) * data.size(1)

#         optimizerG.zero_grad()
#         optimizerE.zero_grad()
#         loss.backward()
#         optimizerG.step()
#         optimizerE.step()
#     return math.exp(total_loss / total_words)

# gen_criterion = nn.NLLLoss(reduction='sum')
# ##generator_optim = optim.Adam(generator.parameters(),lr=0.01)
# #embedding_optim = optim.Adam(embedding.parameters(),lr=0.01)
# if use_cuda:
#     gen_criterion = gen_criterion.cuda()
# print('Pretrain with MLE ...')
# for epoch in range(PRE_EPOCH_NUM):
#     loss = train_epoch(generator, train_loader, gen_criterion, generator_optim, embedding_optim)
#     print('Epoch [%d] Model Loss: %f'% (epoch, loss))

#     z = torch.randn((1, bs, latant_dim)).cuda()
#     # z = torch.zeros((1, bs, hidden_dim)).cuda()
#     samples = sample_from_generator(generator, embedding,bs,50,z,use_mvnrom=True)
#     print(' '.join([TEXT.vocab.itos[i] for i in samples[0]]))
# %%
embedding_optim = optim.Adam(embedding.parameters(),lr=1e-5)
generator_optim = optim.Adam(generator.parameters(),lr=1e-5)
extractor_optim = optim.Adam(extractor.parameters(),lr=1e-5)

embedding_scheduler = optim.lr_scheduler.ExponentialLR(embedding_optim, 0.99)
generator_scheduler = optim.lr_scheduler.ExponentialLR(generator_optim, 0.99)
extractor_scheduler = optim.lr_scheduler.ExponentialLR(extractor_optim, 0.99)
EPOCH = 200
uidx = 0
dis_steps = 3
gen_steps = 1
for epoch in range(EPOCH):
    # train extractor
    losses_disc = []
    losses_gen = []
    kf, set_len = get_minibatches_idx(len(train), BATCH_SIZE, shuffle=True)
    for _, train_index in tqdm(kf,total=set_len):
        uidx += 1
        sents = [train[t] for t in train_index]
        text = torch.tensor(np.stack(sents)).long()

        z = torch.randn((1, text.size(0), latant_dim))
        if use_cuda:
            text = text.cuda()
            z = z.cuda()

        samples = sample_from_generator_soft(generator, embedding,text.size(0),SEQ_LEN,z)

        fake_feat = extractor(embedding.forward_from_vocab_size(samples))
        real_feat = extractor(embedding(text))
        loss = IPOT_distance(fake_feat, real_feat)

        if uidx % dis_steps==0:
            loss_disc = -loss
            extractor_optim.zero_grad()
            loss.backward()
            extractor_optim.step()
            losses_disc.append(loss_disc.item())
            continue

        if uidx % gen_steps==0:
            loss_gen = loss
            embedding_optim.zero_grad()
            generator_optim.zero_grad()
            loss.backward()
            embedding_optim.step()
            generator_optim.step()
            losses_gen.append(loss_gen.item())
            continue

    extractor_scheduler.step()
    embedding_scheduler.step()
    generator_scheduler.step()

    print('Epoch [%d], IPOT_distance Discriminator: %f' % (epoch, -np.mean(losses_disc)))
    print('Epoch [%d], IPOT_distance Generator: %f' % (epoch, np.mean(losses_gen)))

    z = torch.randn((1, 1, latant_dim)).cuda()
    samples = sample_from_generator(generator, embedding,1,50,z,use_mvnrom=False)
    print(' '.join([ixtoword[i.item()] for i in samples[0]]))